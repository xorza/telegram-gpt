#![allow(unused_imports)]
#![allow(dead_code)]

mod conversation;
mod db;
mod openrouter_api;
mod panic_handler;
mod typing;

use anyhow::{Context, anyhow};
use conversation::{Conversation, MessageRole};
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
use tokio::sync::{MappedMutexGuard, Mutex, MutexGuard, RwLock};
use typing::TypingIndicator;

const DEFAULT_MODEL: &str = "xiaomi/mimo-v2-flash:free";
const TELEGRAM_MAX_MESSAGE_LENGTH: usize = 4096;
const STREAM_RESPONSE: bool = false;

#[derive(Debug, Clone)]
struct App {
    bot: Bot,
    http_client: reqwest::Client,
    model: openrouter_api::ModelSummary,
    models: Arc<RwLock<Vec<openrouter_api::ModelSummary>>>,
    conversations: Arc<Mutex<HashMap<ChatId, Conversation>>>,
    db: Arc<Mutex<Connection>>,
    system_prompt0: conversation::Message,
}

#[tokio::main]
async fn main() {
    let app = init().await;

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
}

async fn init() -> App {
    // Ensure all panics are logged once and exit cleanly.
    panic_handler::set_panic_hook();

    dotenv::dotenv().ok();

    // Log to rotating files capped at 10MB each, keeping the 3 newest, while also duplicating info logs to stdout.
    Logger::try_with_env_or_str("info")
        .expect("failed to initialize logger")
        .log_to_file(FileSpec::default().directory("logs"))
        .rotate(
            Criterion::Size(10 * 1024 * 1024),
            Naming::Numbers,
            Cleanup::KeepLogFiles(3),
        )
        .duplicate_to_stdout(Duplicate::All)
        .start()
        .expect("failed to start logger");

    let bot = Bot::from_env();
    let http_client = reqwest::Client::new();
    let model_id = std::env::var("OPENROUTER_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let model = openrouter_api::model(&http_client, &model_id)
        .await
        .expect("failed to load model");
    let models = spawn_model_refresh(http_client.clone()).await;

    let db = Arc::new(Mutex::new(db::init_db()));

    let conversations: Arc<Mutex<HashMap<ChatId, Conversation>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let system_text0 = "Do not output markdown, use plain text.".to_string();
    let system_prompt0 = conversation::Message {
        role: conversation::MessageRole::System,
        text: system_text0,
    };

    log::info!(
        "starting tggpt bot with model: {}, max prompt tokens: {}",
        model.id,
        model.context_length
    );

    App {
        bot,
        http_client,
        model,
        models,
        conversations,
        db,
        system_prompt0,
    }
}

async fn spawn_model_refresh(
    http_client: reqwest::Client,
) -> Arc<RwLock<Vec<openrouter_api::ModelSummary>>> {
    let models = Arc::new(RwLock::new(Vec::new()));

    // Fetch helper keeps the refresh logic in one place.
    async fn refresh_models(
        http_client: &reqwest::Client,
        models: &Arc<RwLock<Vec<openrouter_api::ModelSummary>>>,
    ) {
        match openrouter_api::list_models(http_client).await {
            Ok(latest) => {
                let mut guard = models.write().await;
                *guard = latest;
            }
            Err(err) => {
                log::warn!("failed to refresh model list: {err}");
            }
        }
    }

    // Run once immediately so callers have data on startup.
    refresh_models(&http_client, &models).await;

    let models_clone = models.clone();
    let http_client = http_client.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10 * 60));
        loop {
            interval.tick().await;
            refresh_models(&http_client, &models_clone).await;
        }
    });

    models
}

impl App {
    async fn process_message(&self, msg: Message) -> anyhow::Result<()> {
        let chat_id = msg.chat.id;
        let user_message = self.extract_user_message(chat_id, &msg).await?;

        log::info!("received message from chat {chat_id}");

        let (payload, openai_api_key) = match self.prepare_llm_request(chat_id, &user_message).await
        {
            Ok(ready) => (ready.payload, ready.openai_api_key),
            Err(LlmRequestError::Unauthorized) => {
                let message = format!(
                    "You are not authorized to use this bot. Chat id {}",
                    chat_id
                );
                self.bot.send_message(chat_id, &message).await?;
                return Err(anyhow::anyhow!("Unauthorized"));
            }
            Err(LlmRequestError::NoApiKeyProvided) => {
                let message = format!("No API key provided for chat id {}", chat_id);
                self.bot.send_message(chat_id, &message).await?;
                return Err(anyhow::anyhow!("No API key provided"));
            }
        };

        let typing_indicator = TypingIndicator::new(self.bot.clone(), chat_id);

        let on_stream_delta = {
            let bot = self.bot.clone();
            let stream_buffer = Arc::new(tokio::sync::Mutex::new(String::new()));

            move |delta: String, finalize| {
                handle_stream_delta(bot.clone(), chat_id, stream_buffer.clone(), delta, finalize)
            }
        };

        let llm_response = openrouter_api::send(
            &self.http_client,
            &openai_api_key,
            payload,
            STREAM_RESPONSE,
            on_stream_delta,
        )
        .await;

        drop(typing_indicator);

        match llm_response {
            Ok(llm_response) => {
                let assistant_message = conversation::Message {
                    role: MessageRole::Assistant,
                    text: llm_response.completion_text,
                };
                let messages = [user_message, assistant_message];
                self.persist_messages(chat_id, &messages).await;
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

    async fn extract_user_message(
        &self,
        chat_id: ChatId,
        msg: &Message,
    ) -> anyhow::Result<conversation::Message> {
        let user_text = match msg.text() {
            Some(t) => t.to_owned(),
            None => {
                self.bot
                    .send_message(chat_id, "Only text messages are supported.")
                    .await?;
                return Err(anyhow::anyhow!("Only text messages are supported."));
            }
        };

        Ok(conversation::Message {
            role: MessageRole::User,
            text: user_text,
        })
    }

    async fn prepare_llm_request(
        &self,
        chat_id: ChatId,
        user_message: &conversation::Message,
    ) -> LlmRequestResult {
        let mut conversation = self.get_conversation(chat_id).await;
        if !conversation.is_authorized {
            log::warn!("Unauthorized user {}", chat_id);
            return Err(LlmRequestError::Unauthorized);
        }

        let reserved_tokens = openrouter_api::estimate_tokens([
            self.system_prompt0.text.as_str(),
            conversation
                .system_prompt
                .as_ref()
                .map(|s| s.text.as_str())
                .unwrap_or(""),
            user_message.text.as_str(),
        ]);

        let budget = self
            .model
            .context_length
            .saturating_sub(reserved_tokens + self.model.max_completion_tokens);
        conversation.prune_to_token_budget(budget);

        let mut history = Vec::new();
        history.push(self.system_prompt0.clone());
        if let Some(system_prompt) = conversation.system_prompt.as_ref() {
            history.push(system_prompt.clone());
        }
        history.extend(conversation.history.iter().cloned());
        history.push(user_message.clone());

        let Some(openai_api_key) = conversation.openrouter_api_key.clone() else {
            log::warn!("No API key provided for chat id {}", chat_id);
            return Err(LlmRequestError::NoApiKeyProvided);
        };
        drop(conversation);

        let payload =
            openrouter_api::prepare_payload(&self.model.id, history.iter(), STREAM_RESPONSE);

        Ok(LlmRequestReady {
            payload,
            openai_api_key,
        })
    }

    async fn persist_messages(&self, chat_id: ChatId, messages: &[conversation::Message]) {
        {
            let mut conversation = self.get_conversation(chat_id).await;
            conversation.add_messages(messages.iter().cloned());
        }

        db::add_messages(&self.db, chat_id, messages.iter().cloned()).await;
    }

    async fn get_conversation(&self, chat_id: ChatId) -> MappedMutexGuard<'_, Conversation> {
        let mut conv_map = self.conversations.lock().await;

        if let std::collections::hash_map::Entry::Vacant(entry) = conv_map.entry(chat_id) {
            let conv = db::load_conversation(
                &self.db,
                chat_id,
                self.model
                    .context_length
                    .saturating_sub(self.model.max_completion_tokens),
            )
            .await;
            entry.insert(conv);
        }

        MutexGuard::map(conv_map, |map| {
            map.get_mut(&chat_id).expect("conversation must exist")
        })
    }
}

#[derive(Debug)]
struct LlmRequestReady {
    payload: serde_json::Value,
    openai_api_key: String,
}

#[derive(Debug)]
enum LlmRequestError {
    Unauthorized,
    NoApiKeyProvided,
}

type LlmRequestResult = Result<LlmRequestReady, LlmRequestError>;

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
        for (char_idx, (i, ch)) in buf.char_indices().enumerate() {
            if char_idx == max_chars {
                byte_split = i;
                break;
            }
            if ch.is_whitespace() {
                last_whitespace = Some(i);
            }
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
