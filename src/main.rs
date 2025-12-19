#![allow(unused_imports)]

mod conversation;
mod db;
mod openrouter_api;
mod typing;

use anyhow::{Context, anyhow};
use conversation::{Conversation, MessageRole, TokenCounter};
use flexi_logger::{Cleanup, Criterion, Duplicate, FileSpec, Logger, Naming};
use reqwest::header::PROXY_AUTHENTICATE;
use rusqlite::Connection;
use serde_json::Value;
use std::clone;
use std::fmt::Debug;
use std::{collections::HashMap, path::Path, sync::Arc};
use teloxide::RequestError;
use teloxide::types::CopyTextButton;
use teloxide::{
    prelude::*,
    types::{ChatId, MessageId, ReactionType},
};
use tokio::sync::{MappedMutexGuard, Mutex, MutexGuard};
use typing::TypingIndicator;

const DEFAULT_MODEL: &str = "xiaomi/mimo-v2-flash:free";
const TELEGRAM_MAX_MESSAGE_LENGTH: usize = 4096;
const STREAM_RESPONSE: bool = false;

#[derive(Clone)]
struct App {
    bot: Bot,
    http_client: Arc<reqwest::Client>,
    model: String,
    tokenizer: Arc<TokenCounter>,
    max_prompt_tokens: usize,
    conversations: Arc<Mutex<HashMap<ChatId, Conversation>>>,
    db: Arc<Mutex<Connection>>,
    developer_prompt0: conversation::Message,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let app = init().await?;

    teloxide::repl(app.bot.clone(), move |_bot: Bot, msg: Message| {
        let app = app.clone();
        async move {
            let result = app.process_message(msg).await;

            if let Err(err) = result {
                log::error!("Error processing message: {}", err);
            }

            respond(())
        }
    })
    .await;

    Ok(())
}

async fn init() -> anyhow::Result<App, anyhow::Error> {
    dotenv::dotenv().ok();

    // Log to rotating files capped at 10MB each, keeping the 3 newest, while also duplicating info logs to stdout.
    Logger::try_with_env_or_str("info")?
        .log_to_file(FileSpec::default().directory("logs"))
        .rotate(
            Criterion::Size(10 * 1024 * 1024),
            Naming::Numbers,
            Cleanup::KeepLogFiles(3),
        )
        .duplicate_to_stdout(Duplicate::All)
        .start()?;

    let bot = Bot::from_env();
    let http_client = Arc::new(reqwest::Client::new());
    let model = std::env::var("OPENROUTER_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let tokenizer = Arc::new(TokenCounter::new(&model));

    let max_prompt_tokens = openai_api::context_length(&model);

    let db = Arc::new(Mutex::new(db::init_db()?));

    let conversations: Arc<Mutex<HashMap<ChatId, Conversation>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let developer_text0 = "Do not output markdown, use plain text.".to_string();
    let developer_prompt0 = conversation::Message {
        role: conversation::MessageRole::Developer,
        tokens: tokenizer.count_text(&developer_text0),
        text: developer_text0,
    };

    log::info!("starting tggpt bot with model: {model}, max prompt tokens: {max_prompt_tokens}");

    Ok(App {
        bot,
        http_client,
        model,
        tokenizer,
        max_prompt_tokens,
        conversations,
        db,
        developer_prompt0,
    })
}

impl App {
    async fn process_message(&self, msg: Message) -> anyhow::Result<()> {
        let chat_id = msg.chat.id;
        let user_text = match msg.text() {
            Some(t) => t.to_owned(),
            None => {
                self.bot
                    .send_message(chat_id, "Only text messages are supported.")
                    .await?;
                return Ok(());
            }
        };
        let user_message =
            conversation::Message::with_text(MessageRole::User, user_text, &self.tokenizer);

        let typing_indicator = TypingIndicator::new(self.bot.clone(), chat_id);

        log::info!("received message from chat {chat_id}");

        let mut conversation = self.get_conversation(chat_id).await?;
        if !conversation.is_authorized {
            log::warn!("Unauthorized user {}", chat_id);

            let message = format!(
                "You are not authorized to use this bot. Chat id {}",
                chat_id
            );
            self.bot.send_message(msg.chat.id, &message).await?;
            return Ok(());
        }

        let developer_prompt_tokens = self.developer_prompt0.tokens
            + conversation
                .developer_prompt
                .as_ref()
                .map(|p| p.tokens)
                .unwrap_or(0);
        let budget = self
            .max_prompt_tokens
            .saturating_sub(developer_prompt_tokens + user_message.tokens);
        conversation.prune_to_token_budget(budget);

        let history = std::iter::once(&self.developer_prompt0)
            .chain(conversation.developer_prompt.iter())
            .chain(conversation.history.iter())
            .chain(std::iter::once(&user_message));

        let payload = openrouter_api::prepare_payload(&self.model, history, STREAM_RESPONSE);
        let openai_api_key = conversation.openai_api_key.clone();

        drop(conversation);

        let on_stream_delta = {
            let bot = self.bot.clone();
            let stream_buffer = Arc::new(tokio::sync::Mutex::new(String::new()));

            move |delta: String, finalize| {
                handle_stream_delta(bot.clone(), chat_id, stream_buffer.clone(), delta, finalize)
            }
        };

        let llm_result = openai_api::send(
            &self.http_client,
            &openai_api_key,
            payload,
            STREAM_RESPONSE,
            on_stream_delta,
        )
        .await;

        drop(typing_indicator);

        match llm_result {
            Ok(assistant_text) => {
                let assistant_message = conversation::Message::with_text(
                    MessageRole::Assistant,
                    assistant_text,
                    &self.tokenizer,
                );
                let messages = [user_message, assistant_message];
                self.get_conversation(chat_id)
                    .await?
                    .add_messages(messages.iter().cloned());
                db::add_messages(&self.db, chat_id, messages.into_iter()).await?;
            }
            Err(err) => {
                log::error!("failed to get llm response: {err}");

                self.bot
                    .set_message_reaction(chat_id, msg.id)
                    .reaction(vec![ReactionType::Emoji {
                        emoji: "🖕".to_string(),
                    }])
                    .await?;
            }
        }

        Ok(())
    }

    async fn get_conversation(
        &self,
        chat_id: ChatId,
    ) -> anyhow::Result<MappedMutexGuard<'_, Conversation>> {
        let mut conv_map = self.conversations.lock().await;

        if !conv_map.contains_key(&chat_id) {
            let conv =
                db::load_conversation(&self.db, chat_id, &self.tokenizer, self.max_prompt_tokens)
                    .await?;
            conv_map.insert(chat_id, conv);
        }

        Ok(MutexGuard::map(conv_map, |map| {
            map.get_mut(&chat_id).expect("conversation must exist")
        }))
    }
}

impl Debug for App {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("App")
            .field("bot", &self.bot)
            .field("http_client", &self.http_client)
            .field("model", &self.model)
            .field("tokenizer", &"?")
            .field("max_prompt_tokens", &self.max_prompt_tokens)
            .field("conversations", &self.conversations)
            .finish()
    }
}

async fn handle_stream_delta(
    bot: Bot,
    chat_id: ChatId,
    stream_buffer: Arc<tokio::sync::Mutex<String>>,
    delta: String,
    finalize: bool,
) -> anyhow::Result<()> {
    // Accumulate incoming delta into the shared buffer.
    let mut buf = stream_buffer.lock().await;
    buf.push_str(&delta);

    // Remove and return up to `max_chars` characters, preferring to split on whitespace.
    fn take_prefix(buf: &mut String, max_chars: usize) -> String {
        let mut last_whitespace = None;
        let mut byte_split = buf.len();
        let mut char_idx = 0usize;

        for (i, ch) in buf.char_indices() {
            if char_idx == max_chars {
                byte_split = i;
                break;
            }
            if ch.is_whitespace() {
                last_whitespace = Some(i);
            }
            char_idx += 1;
        }

        // Buffer shorter than max_chars — take it all.
        if byte_split == buf.len() {
            return std::mem::take(buf);
        }

        // Prefer splitting on the last whitespace before the limit (unless it would be empty).
        let split_at = last_whitespace.filter(|idx| *idx > 0).unwrap_or(byte_split);
        let tail = buf.split_off(split_at);
        std::mem::replace(buf, tail)
    }

    // Send full Telegram-sized chunks as soon as they accumulate.
    while buf.chars().count() >= TELEGRAM_MAX_MESSAGE_LENGTH {
        let chunk = take_prefix(&mut buf, TELEGRAM_MAX_MESSAGE_LENGTH);
        let to_send = chunk.clone();
        drop(buf);
        bot.send_message(chat_id, to_send).await?;
        buf = stream_buffer.lock().await;
    }

    // On final signal, flush any remainder.
    if finalize && !buf.is_empty() {
        let to_send = std::mem::take(&mut *buf);
        drop(buf);
        bot.send_message(chat_id, to_send).await?;
    }

    Ok(())
}
