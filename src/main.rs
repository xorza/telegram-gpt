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
use teloxide::utils::command::BotCommands;
use teloxide::{
    prelude::*,
    types::{ChatId, MessageId, ReactionType},
};
use tokio::sync::{MappedMutexGuard, Mutex, MutexGuard, RwLock};
use typing::TypingIndicator;

const DEFAULT_MODEL_FALLBACK: &str = "xiaomi/mimo-v2-flash:free";
const TELEGRAM_MAX_MESSAGE_LENGTH: usize = 4096;

#[derive(Debug, Clone)]
struct App {
    bot: Bot,
    bot_username: String,
    http_client: reqwest::Client,
    models: Arc<RwLock<Vec<openrouter_api::ModelSummary>>>,
    conversations: Arc<Mutex<HashMap<ChatId, Conversation>>>,
    db: Arc<Mutex<Connection>>,
    system_prompt0: conversation::Message,
    default_model: String,
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
    let bot_username = bot
        .get_me()
        .await
        .expect("failed to fetch bot user info")
        .user
        .username
        .unwrap_or_default();
    let models = spawn_model_refresh(http_client.clone()).await;
    let db = Arc::new(Mutex::new(db::init_db()));
    let conversations: Arc<Mutex<HashMap<ChatId, Conversation>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let system_text0 = "Do not output markdown, use plain text.".to_string();
    let system_prompt0 = conversation::Message {
        role: conversation::MessageRole::System,
        text: system_text0,
    };
    let default_model =
        std::env::var("DEFAULT_MODEL").unwrap_or_else(|_| DEFAULT_MODEL_FALLBACK.to_string());

    log::info!("starting tggpt bot");

    App {
        bot,
        bot_username,
        http_client,
        models,
        conversations,
        db,
        system_prompt0,
        default_model,
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

        self.maybe_update_user_name(&msg).await;

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
        let mut chunk = String::new();
        let mut chunk_len = 0usize;

        for token in text.split_inclusive([' ', '\n']) {
            let token_len = token.chars().count();
            if token_len > TELEGRAM_MAX_MESSAGE_LENGTH {
                if !buffer.is_empty() {
                    self.bot.send_message(chat_id, &buffer).await?;
                    buffer.clear();
                    buffer_len = 0;
                }
                for ch in token.chars() {
                    if chunk_len == TELEGRAM_MAX_MESSAGE_LENGTH {
                        self.bot.send_message(chat_id, &chunk).await?;
                        chunk.clear();
                        chunk_len = 0;
                    }
                    chunk.push(ch);
                    chunk_len += 1;
                }
                if !chunk.is_empty() {
                    self.bot.send_message(chat_id, &chunk).await?;
                    chunk.clear();
                    chunk_len = 0;
                }
                continue;
            }
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

    async fn maybe_update_user_name(&self, msg: &Message) {
        let Some(user) = msg.from.as_ref() else {
            return;
        };

        let user_name = user.username.clone().or_else(|| {
            let mut name = user.first_name.clone();
            if let Some(last) = user.last_name.as_ref()
                && !last.is_empty()
            {
                if !name.is_empty() {
                    name.push(' ');
                }
                name.push_str(last);
            }
            if name.is_empty() { None } else { Some(name) }
        });

        let Some(user_name) = user_name else {
            return;
        };

        let chat_id = msg.chat.id;
        let (should_update, old_name) = {
            let mut conv = self.get_conversation(chat_id).await;
            if conv.user_name.as_deref() != Some(user_name.as_str()) {
                let old_name = conv.user_name.clone();
                conv.user_name = Some(user_name.clone());
                (true, old_name)
            } else {
                (false, None)
            }
        };

        if should_update {
            log::info!(
                "Updating user name for chat {}: {:?} -> {:?}",
                chat_id,
                old_name,
                user_name
            );
            db::set_user_name(&self.db, chat_id, Some(&user_name)).await;
        }
    }

    async fn process_command(
        &self,
        chat_id: ChatId,
        user_message: &conversation::Message,
    ) -> anyhow::Result<()> {
        let command = match parse_command(user_message.text.as_str(), &self.bot_username) {
            Ok(Some(command)) => command,
            Ok(None) => {
                // Command addressed to a different bot; ignore silently.
                return Ok(());
            }
            Err(message) => {
                log::warn!("Failed to parse command: {}", message);
                self.bot.send_message(chat_id, message).await?;
                return Ok(());
            }
        };

        log::info!("Received command: {:?}", command);
        match command {
            Command::Help | Command::Start => {
                let message = [
                    "Commands:",
                    "/help - show this help",
                    "/start - show this help",
                    "/models - list available models",
                    "/model [id|none] - show or set model",
                    "/key [key|none] - show or set API key",
                    "/systemprompt [text|none] - show or set system prompt",
                    "/approve [chat_id true|false] - admin only",
                ]
                .join("\n");
                self.bot_split_send(chat_id, &message).await?;
            }
            Command::Models => {
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
            Command::Model(arg) => match arg {
                CommandArg::Empty => {
                    let current_model_id = {
                        let conv = self.get_conversation(chat_id).await;
                        conv.model_id.clone()
                    };
                    let model = self.resolve_model(current_model_id.as_deref()).await;
                    self.bot
                        .send_message(chat_id, format!("Current model: `{}`", model.id))
                        .await?;
                }
                CommandArg::None => {
                    {
                        let mut conv = self.get_conversation(chat_id).await;
                        conv.model_id = None;
                    }
                    db::set_model_id(&self.db, chat_id, None).await;
                    self.bot
                        .send_message(chat_id, "Model cleared; using default.")
                        .await?;
                }
                CommandArg::Text(model_id) => {
                    let available_models = self.models.read().await;
                    let selected_model = available_models.iter().find(|m| m.id == model_id);

                    if let Some(model) = selected_model {
                        {
                            let mut conv = self.get_conversation(chat_id).await;
                            conv.model_id = Some(model.id.clone());
                        }
                        db::set_model_id(&self.db, chat_id, Some(&model.id)).await;
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
            },
            Command::Key(arg) => match arg {
                CommandArg::Empty => {
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
                }
                CommandArg::None => {
                    {
                        let mut conv = self.get_conversation(chat_id).await;
                        conv.openrouter_api_key = None;
                    }
                    db::set_openrouter_api_key(&self.db, chat_id, None).await;
                    self.bot.send_message(chat_id, "API key cleared.").await?;
                }
                CommandArg::Text(key) => {
                    {
                        let mut conv = self.get_conversation(chat_id).await;
                        conv.openrouter_api_key = Some(key.clone());
                    }
                    db::set_openrouter_api_key(&self.db, chat_id, Some(&key)).await;
                    self.bot.send_message(chat_id, "API key updated.").await?;
                }
            },
            Command::SystemPrompt(arg) => match arg {
                CommandArg::Empty => {
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
                }
                CommandArg::None => {
                    {
                        let mut conv = self.get_conversation(chat_id).await;
                        conv.system_prompt = None;
                    }
                    db::set_system_prompt(&self.db, chat_id, None).await;
                    self.bot
                        .send_message(chat_id, "System prompt cleared.")
                        .await?;
                }
                CommandArg::Text(prompt) => {
                    {
                        let mut conv = self.get_conversation(chat_id).await;
                        conv.system_prompt = Some(conversation::Message {
                            role: MessageRole::System,
                            text: prompt.clone(),
                        });
                    }
                    db::set_system_prompt(&self.db, chat_id, Some(&prompt)).await;
                    self.bot
                        .send_message(chat_id, "System prompt updated.")
                        .await?;
                }
            },
            Command::Approve(approve) => {
                let is_admin = { self.get_conversation(chat_id).await.is_admin };
                if !is_admin {
                    self.bot
                        .send_message(chat_id, "You are not authorized to use /approve.")
                        .await?;
                    return Ok(());
                }

                match approve {
                    ApproveArg::Empty => {
                        let pending = db::list_unauthorized_chats(&self.db).await;
                        if pending.is_empty() {
                            self.bot.send_message(chat_id, "No pending users.").await?;
                            return Ok(());
                        }

                        let mut lines = Vec::with_capacity(pending.len());
                        for (pending_id, name) in pending {
                            let display_name = name.unwrap_or_else(|| "unknown".to_string());
                            lines.push(format!("{} - {}", pending_id, display_name));
                        }

                        let message = format!("Pending users:\n{}", lines.join("\n"));
                        self.bot_split_send(chat_id, &message).await?;
                    }
                    ApproveArg::ApproveChat {
                        chat_id: target_chat_id,
                        is_authorized,
                    } => {
                        let target_id = ChatId(target_chat_id as i64);
                        db::set_is_authorized(&self.db, target_id, is_authorized).await;

                        let mut conv_map = self.conversations.lock().await;
                        if let Some(conv) = conv_map.get_mut(&target_id) {
                            conv.is_authorized = is_authorized;
                        }

                        let message = format!(
                            "Updated chat {} is_authorized={}",
                            target_chat_id, is_authorized
                        );
                        self.bot.send_message(chat_id, message).await?;
                    }
                    ApproveArg::Invalid => {
                        self.bot
                            .send_message(chat_id, "Usage: /approve <chat_id> <true|false>")
                            .await?;
                    }
                }
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

        let payload = openrouter_api::prepare_payload(&model.id, history.iter(), false);

        Ok(LlmRequestReady {
            payload,
            openrouter_api_key: openai_api_key,
        })
    }

    async fn resolve_model(&self, model_id: Option<&str>) -> openrouter_api::ModelSummary {
        let requested = model_id.unwrap_or(self.default_model.as_str());
        let models = self.models.read().await;
        models
            .iter()
            .find(|m| m.id == requested)
            .cloned()
            .or_else(|| {
                models
                    .iter()
                    .find(|m| m.id == self.default_model.as_str())
                    .cloned()
            })
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
enum CommandArg {
    Empty,
    None,
    Text(String),
}

impl CommandArg {
    fn from_text(text: Option<&str>) -> Self {
        match text {
            Some(text) => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    CommandArg::Empty
                } else if trimmed.eq_ignore_ascii_case("none") {
                    CommandArg::None
                } else {
                    CommandArg::Text(trimmed.to_string())
                }
            }
            None => CommandArg::Empty,
        }
    }
}

#[derive(Debug)]
enum Command {
    /// Show this help text.
    Help,
    /// Show this help text.
    Start,
    /// List available models.
    Models,
    /// Get/set the model (use `none` to clear).
    Model(CommandArg),
    /// Get/set the API key (use `none` to clear).
    Key(CommandArg),
    /// Get/set the system prompt (use `none` to clear).
    SystemPrompt(CommandArg),
    /// List or update chat authorization.
    Approve(ApproveArg),
}

#[derive(Debug)]
enum ApproveArg {
    Empty,
    Invalid,
    ApproveChat { chat_id: u64, is_authorized: bool },
}

fn parse_command(text: &str, bot_username: &str) -> Result<Option<Command>, String> {
    let trimmed = text.trim();
    if !trimmed.starts_with('/') {
        return Err("Unknown command".to_string());
    }

    let without_slash = trimmed.trim_start_matches('/');
    let (cmd_part, args_part) = match without_slash.find(char::is_whitespace) {
        Some(idx) => (
            &without_slash[..idx],
            Some(without_slash[idx..].trim_start()),
        ),
        None => (without_slash, None),
    };
    let args_part = args_part.and_then(|args| if args.is_empty() { None } else { Some(args) });

    let (cmd_name, mention) = match cmd_part.split_once('@') {
        Some((cmd, mention)) => (cmd, Some(mention)),
        None => (cmd_part, None),
    };

    if let Some(mention) = mention {
        if !mention.eq_ignore_ascii_case(bot_username) {
            return Ok(None);
        }
    }

    match cmd_name.to_ascii_lowercase().as_str() {
        "help" => {
            if args_part.is_none() {
                Ok(Some(Command::Help))
            } else {
                Err("Unknown command".to_string())
            }
        }
        "start" => {
            if args_part.is_none() {
                Ok(Some(Command::Start))
            } else {
                Err("Unknown command".to_string())
            }
        }
        "models" => {
            if args_part.is_none() {
                Ok(Some(Command::Models))
            } else {
                Err("Unknown command".to_string())
            }
        }
        "model" => Ok(Some(Command::Model(CommandArg::from_text(args_part)))),
        "key" => Ok(Some(Command::Key(CommandArg::from_text(args_part)))),
        "systemprompt" => Ok(Some(Command::SystemPrompt(CommandArg::from_text(
            args_part,
        )))),
        "approve" => {
            if args_part.is_none() {
                return Ok(Some(Command::Approve(ApproveArg::Empty)));
            }
            let args = args_part.unwrap().split_whitespace().collect::<Vec<&str>>();
            if args.len() != 2 {
                return Ok(Some(Command::Approve(ApproveArg::Invalid)));
            }

            let chat_id: u64 = match args[0].parse() {
                Ok(value) => value,
                Err(_) => {
                    return Ok(Some(Command::Approve(ApproveArg::Invalid)));
                }
            };
            let is_authorized = match args[1].to_ascii_lowercase().as_str() {
                "true" | "1" => true,
                "false" | "0" => false,
                _ => {
                    return Ok(Some(Command::Approve(ApproveArg::Invalid)));
                }
            };
            Ok(Some(Command::Approve(ApproveArg::ApproveChat {
                chat_id,
                is_authorized,
            })))
        }
        _ => Err("Unknown command".to_string()),
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
