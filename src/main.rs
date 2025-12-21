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
    let models = spawn_model_refresh(http_client.clone()).await;
    let db = Arc::new(Mutex::new(db::init_db()));
    let conversations: Arc<Mutex<HashMap<ChatId, Conversation>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let system_text0 = "Do not output markdown, use plain text.".to_string();
    let system_prompt0 = conversation::Message {
        role: conversation::MessageRole::System,
        text: system_text0,
    };

    log::info!("starting tggpt bot");

    App {
        bot,
        http_client,
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
    ) -> anyhow::Result<()> {
        let latest = openrouter_api::list_models(http_client).await?;

        let mut guard = models.write().await;
        *guard = latest;

        Ok(())
    }

    // Run once immediately; keep retrying so we always start with a model list.
    let mut attempt = 1u32;
    loop {
        match refresh_models(&http_client, &models).await {
            Ok(()) => break,
            Err(err) => {
                log::warn!(
                    "initial model fetch failed (attempt {}): {err}; retrying in 5s",
                    attempt
                );
                attempt += 1;
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        }
    }

    let models_clone = models.clone();
    let http_client = http_client.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10 * 60));
        loop {
            interval.tick().await;
            refresh_models(&http_client, &models_clone).await.ok();
        }
    });

    models
}

impl App {
    async fn process_message(&self, msg: Message) -> anyhow::Result<()> {
        let chat_id = msg.chat.id;
        log::info!("received message from chat {}", chat_id);

        if !self.get_conversation(chat_id).await.is_authorized {
            let message = format!(
                "You are not authorized to use this bot. Chat id {}",
                chat_id
            );
            self.bot.send_message(chat_id, &message).await?;
            return Err(anyhow::anyhow!("Unauthorized"));
        }

        let user_message = self.extract_user_message(chat_id, &msg).await?;
        if user_message.text.starts_with("/") {
            self.process_command(chat_id, &user_message).await?;
            return Ok(());
        }

        let (payload, openai_api_key) = match self.prepare_llm_request(chat_id, &user_message).await
        {
            Ok(ready) => (ready.payload, ready.openrouter_api_key),
            Err(LlmRequestError::NoApiKeyProvided) => {
                let message = format!("No API key provided for chat id {}", chat_id);
                self.bot.send_message(chat_id, &message).await?;
                return Err(anyhow::anyhow!("No API key provided"));
            }
        };

        let llm_response = {
            let _typing_indicator = TypingIndicator::new(self.bot.clone(), chat_id);
            openrouter_api::send(&self.http_client, &openai_api_key, payload).await
        };

        match llm_response {
            Ok(llm_response) => {
                self.bot_split_send(chat_id, &llm_response.completion_text)
                    .await?;
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

    async fn bot_split_send(&self, chat_id: ChatId, text: &str) -> anyhow::Result<()> {
        if text.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH {
            self.bot.send_message(chat_id, text).await?;
            return Ok(());
        }

        let mut buffer = String::new();
        let mut buffer_len = 0usize;

        for token in text.split_inclusive(|c| c == ' ' || c == '\n') {
            let token_len = token.chars().count();
            if buffer_len + token_len > TELEGRAM_MAX_MESSAGE_LENGTH && !buffer.is_empty() {
                self.bot.send_message(chat_id, &buffer).await?;
                buffer.clear();
                buffer_len = 0;
            }

            buffer.push_str(token);
            buffer_len += token_len;
        }

        if !buffer.is_empty() {
            self.bot.send_message(chat_id, &buffer).await?;
        }

        Ok(())
    }

    async fn process_command(
        &self,
        chat_id: ChatId,
        user_message: &conversation::Message,
    ) -> anyhow::Result<()> {
        let command = user_message.text.split_whitespace().next().unwrap();
        log::info!("Received command: {}", command);
        match command {
            "/models" => {
                let models = self.models.read().await;
                let models = models
                    .iter()
                    .filter_map(|f| {
                        if f.id.starts_with("openai")
                            || f.id.starts_with("anthropic")
                            || f.id.starts_with("google")
                            || f.id.starts_with("x-ai")
                            || f.id.starts_with("deepseek")
                        {
                            Some(f.id.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let message = format!("Available models:\n{}", models);
                self.bot_split_send(chat_id, &message).await?;
            }
            "/model" => {
                let model_id = user_message.text.trim_start_matches("/model").trim();
                if model_id.is_empty() {
                    let current_model_id = {
                        let conv = self.get_conversation(chat_id).await;
                        conv.model_id.clone()
                    };
                    let model = self.resolve_model(current_model_id.as_deref()).await;
                    self.bot
                        .send_message(chat_id, format!("Current model: `{}`", model.id))
                        .await?;
                } else {
                    let available_models = self.models.read().await;
                    let selected_model = available_models.iter().find(|m| m.id == model_id);

                    if let Some(model) = selected_model {
                        {
                            let mut conv = self.get_conversation(chat_id).await;
                            conv.model_id = Some(model.id.clone());
                        }
                        log::info!("User {} selected model: `{}`", chat_id, model.name);
                        self.bot
                            .send_message(chat_id, format!("Selected model: `{}`", model.name))
                            .await?;
                    } else {
                        log::warn!(
                            "User {} tried to select non-existent model: `{}`",
                            chat_id,
                            model_id
                        );
                        self.bot
                            .send_message(chat_id, format!("Model not found: `{}`", model_id))
                            .await?;
                    }
                }
            }
            "/key" => {
                let key = user_message.text.trim_start_matches("/key").trim();
                if key.is_empty() {
                    let current_key = {
                        let conv = self.get_conversation(chat_id).await;
                        conv.openrouter_api_key.clone()
                    };
                    match current_key {
                        Some(key) => {
                            self.bot
                                .send_message(chat_id, format!("Current API key: `{}`", key))
                                .await?;
                        }
                        None => {
                            self.bot.send_message(chat_id, "No API key set.").await?;
                        }
                    }
                } else {
                    {
                        let mut conv = self.get_conversation(chat_id).await;
                        conv.openrouter_api_key = Some(key.to_string());
                    }
                    self.bot.send_message(chat_id, "API key updated.").await?;
                }
            }
            "/system_prompt" => {
                let prompt = user_message
                    .text
                    .trim_start_matches("/system_prompt")
                    .trim();
                if prompt.is_empty() {
                    let current_prompt = {
                        let conv = self.get_conversation(chat_id).await;
                        conv.system_prompt.as_ref().map(|p| p.text.clone())
                    };
                    match current_prompt {
                        Some(prompt) => {
                            self.bot
                                .send_message(
                                    chat_id,
                                    format!("Current system prompt: `{}`", prompt),
                                )
                                .await?;
                        }
                        None => {
                            self.bot
                                .send_message(chat_id, "No system prompt set.")
                                .await?;
                        }
                    }
                } else {
                    {
                        let mut conv = self.get_conversation(chat_id).await;
                        conv.system_prompt = Some(conversation::Message {
                            role: MessageRole::System,
                            text: prompt.to_string(),
                        });
                    }
                    self.bot
                        .send_message(chat_id, "System prompt updated.")
                        .await?;
                }
            }
            _ => {
                self.bot.send_message(chat_id, "Unknown command").await?;
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
            Some(t) => t.trim().to_owned(),
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
        let model = self.resolve_model(conversation.model_id.as_deref()).await;

        let reserved_tokens = openrouter_api::estimate_tokens([
            self.system_prompt0.text.as_str(),
            conversation
                .system_prompt
                .as_ref()
                .map(|s| s.text.as_str())
                .unwrap_or(""),
            user_message.text.as_str(),
        ]);

        conversation.prune_to_token_budget(model.token_budget().saturating_sub(reserved_tokens));

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

        let payload = openrouter_api::prepare_payload(&model.id, history.iter(), STREAM_RESPONSE);

        Ok(LlmRequestReady {
            payload,
            openrouter_api_key: openai_api_key,
        })
    }

    async fn resolve_model(&self, model_id: Option<&str>) -> openrouter_api::ModelSummary {
        let requested = model_id.unwrap_or(DEFAULT_MODEL);
        let models = self.models.read().await;
        models
            .iter()
            .find(|m| m.id == requested)
            .cloned()
            .or_else(|| models.iter().find(|m| m.id == DEFAULT_MODEL).cloned())
            .expect("default model not found")
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
            let mut conversation = db::load_conversation(&self.db, chat_id).await;
            let model = self.resolve_model(conversation.model_id.as_deref()).await;

            db::load_history(&self.db, &mut conversation, model.token_budget()).await;

            log::info!(
                "Loaded conversation {} with {} messages. Model id is {}",
                conversation.chat_id,
                conversation.history.len(),
                model.id
            );

            entry.insert(conversation);
        }

        MutexGuard::map(conv_map, |map| {
            map.get_mut(&chat_id)
                .expect("conversation entry just inserted or already existed")
        })
    }
}

#[derive(Debug)]
struct LlmRequestReady {
    payload: serde_json::Value,
    openrouter_api_key: String,
}

#[derive(Debug)]
enum LlmRequestError {
    NoApiKeyProvided,
}

type LlmRequestResult = Result<LlmRequestReady, LlmRequestError>;
