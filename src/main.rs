#![allow(unused_imports)]

mod conversation;
mod db;
mod openai_api;
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

type DynError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_OPEN_AI_MODEL: &str = "gpt-4.1";
const TELEGRAM_MAX_MESSAGE_LENGTH: usize = 4096;

#[derive(Clone)]
struct App {
    bot: Bot,
    http_client: Arc<reqwest::Client>,
    model: String,
    tokenizer: Arc<TokenCounter>,
    max_prompt_tokens: usize,
    conversations: Arc<Mutex<HashMap<ChatId, Conversation>>>,
    db: Arc<Mutex<Connection>>,
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

struct StreamState {
    full_answer: String,
    buffer: String,
}

struct PreparedRequest {
    user_message: conversation::Message,
    payload: serde_json::Value,
    openai_api_key: String,
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
        .duplicate_to_stdout(Duplicate::Debug)
        .start()?;

    let bot = Bot::from_env();
    let http_client = Arc::new(reqwest::Client::new());
    let model =
        std::env::var("OPEN_AI_MODEL").unwrap_or_else(|_| DEFAULT_OPEN_AI_MODEL.to_string());
    let tokenizer = Arc::new(TokenCounter::new(&model));

    let max_prompt_tokens = openai_api::context_length(&model);

    let db = Arc::new(Mutex::new(db::init_db()?));

    let conversations: Arc<Mutex<HashMap<ChatId, Conversation>>> =
        Arc::new(Mutex::new(HashMap::new()));

    log::info!("starting tggpt bot with model: {model}, max prompt tokens: {max_prompt_tokens}");

    Ok(App {
        bot,
        http_client,
        model,
        tokenizer,
        max_prompt_tokens,
        conversations,
        db,
    })
}

fn split_message(text: &str) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut chunks = Vec::new();
    let mut buffer = String::new();
    let mut buffer_len = 0;

    for ch in text.chars() {
        if buffer_len == TELEGRAM_MAX_MESSAGE_LENGTH {
            chunks.push(std::mem::take(&mut buffer));
            buffer_len = 0;
        }

        buffer.push(ch);
        buffer_len += 1;
    }

    if !buffer.is_empty() {
        chunks.push(buffer);
    }

    chunks
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

impl App {
    async fn process_message(&self, msg: Message) -> anyhow::Result<()> {
        if msg.text().is_none() {
            self.bot
                .send_message(
                    msg.chat.id,
                    "Please send text messages so I can ask the language model.",
                )
                .await?;

            return Ok(());
        }

        let chat_id = msg.chat.id;
        let typing_indicator = TypingIndicator::new(self.bot.clone(), chat_id);

        log::info!("received message from chat {chat_id}");

        let user_text = msg.text().unwrap().to_owned();
        let PreparedRequest {
            user_message,
            payload,
            openai_api_key,
        } = self.prepare_request(chat_id, user_text).await?;

        let streaming = true; // toggle streaming behavior
        let llm_result = self
            .fetch_and_deliver(streaming, chat_id, &openai_api_key, payload)
            .await;

        drop(typing_indicator);

        self.persist_or_react(chat_id, msg.id, user_message, llm_result)
            .await?;

        Ok(())
    }

    async fn prepare_request(
        &self,
        chat_id: ChatId,
        user_text: String,
    ) -> anyhow::Result<PreparedRequest> {
        let mut conversation = self.get_conversation(chat_id).await?;
        if !conversation.is_authorized {
            let error = format!("Unauthorized user {}", chat_id);
            log::warn!("{}", error);
            return Err(anyhow::anyhow!(error));
        }

        let user_message =
            conversation::Message::with_text(MessageRole::User, user_text, &self.tokenizer);

        let system_prompt_tokens = conversation.system_prompt.as_ref().map_or(0, |p| p.tokens);
        conversation.prune_to_token_budget(
            self.max_prompt_tokens - system_prompt_tokens - user_message.tokens,
        );

        let payload = openai_api::prepare_payload(
            &self.model,
            conversation
                .system_prompt
                .as_ref()
                .into_iter()
                .chain(conversation.history.iter())
                .chain(std::iter::once(&user_message)),
        );

        Ok(PreparedRequest {
            user_message,
            payload,
            openai_api_key: conversation.openai_api_key.clone(),
        })
    }

    async fn fetch_and_deliver(
        &self,
        streaming: bool,
        chat_id: ChatId,
        api_key: &str,
        payload: serde_json::Value,
    ) -> anyhow::Result<String, DynError> {
        if !streaming {
            let answer = openai_api::send(&self.http_client, api_key, payload).await?;
            for chunk in split_message(&answer) {
                self.bot.send_message(chat_id, chunk).await?;
            }
            return Ok(answer);
        }

        fn take_prefix(buf: &mut String, max_chars: usize) -> String {
            let mut char_idx = 0usize;
            let mut byte_split = buf.len();
            for (i, _) in buf.char_indices() {
                if char_idx == max_chars {
                    byte_split = i;
                    break;
                }
                char_idx += 1;
            }
            let tail = buf.split_off(byte_split);
            std::mem::replace(buf, tail)
        }

        let state = Arc::new(tokio::sync::Mutex::new(StreamState {
            full_answer: String::new(),
            buffer: String::new(),
        }));

        openai_api::send_stream(&self.http_client, api_key, payload, {
            let bot = self.bot.clone();
            let state = state.clone();
            move |delta| {
                let bot = bot.clone();
                let state = state.clone();
                async move {
                    let mut st = state.lock().await;
                    st.full_answer.push_str(&delta);
                    st.buffer.push_str(&delta);

                    while st.buffer.chars().count() >= TELEGRAM_MAX_MESSAGE_LENGTH {
                        let chunk = take_prefix(&mut st.buffer, TELEGRAM_MAX_MESSAGE_LENGTH);
                        let to_send = chunk.clone();
                        drop(st);
                        bot.send_message(chat_id, to_send).await?;
                        st = state.lock().await;
                    }

                    Ok(())
                }
            }
        })
        .await?;

        // Flush any remaining buffered text after streaming completes.
        {
            let mut st = state.lock().await;
            if !st.buffer.is_empty() {
                let to_send = std::mem::take(&mut st.buffer);
                drop(st);
                self.bot.send_message(chat_id, to_send).await?;
            }
        }

        let answer = state.lock().await.full_answer.clone();
        Ok(answer)
    }

    async fn persist_or_react(
        &self,
        chat_id: ChatId,
        message_id: MessageId,
        user_message: conversation::Message,
        llm_result: anyhow::Result<String, DynError>,
    ) -> anyhow::Result<()> {
        let mut conversation = self.get_conversation(chat_id).await?;

        match llm_result {
            Ok(answer) => {
                let assistant_message = conversation::Message::with_text(
                    MessageRole::Assistant,
                    answer,
                    &self.tokenizer,
                );

                let messages = [user_message, assistant_message];

                conversation.add_messages(messages.iter().cloned());
                db::add_messages(&self.db, chat_id, messages.into_iter()).await?;
            }
            Err(err) => {
                log::error!("failed to get llm response: {err}");

                self.bot
                    .set_message_reaction(chat_id, message_id)
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
