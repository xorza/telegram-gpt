#![allow(unused_imports)]
#![allow(dead_code)]

mod commands;
mod conversation;
mod db;
mod models;
mod openrouter_api;
mod panic_handler;
mod telegram;
mod typing;

use anyhow::{Context, anyhow};
use conversation::{Conversation, MessageRole};
use flexi_logger::{Cleanup, Criterion, Duplicate, FileSpec, Logger, Naming};
use std::{collections::HashMap, sync::Arc};
use telegram::{bot_split_send_formatted, escape_markdown_v2};
use teloxide::{
    prelude::*,
    types::{ChatId, MessageId, MessageKind, ParseMode, ReactionType},
};
use tokio::sync::{MappedMutexGuard, Mutex, MutexGuard, RwLock};
use typing::TypingIndicator;

const DEFAULT_MODEL_FALLBACK: &str = "xiaomi/mimo-v2-flash:free";

#[derive(Debug, Clone)]
struct App {
    bot: Bot,
    bot_username: String,
    http_client: reqwest::Client,
    models: Arc<RwLock<Vec<openrouter_api::ModelSummary>>>,
    conversations: Arc<Mutex<HashMap<ChatId, Conversation>>>,
    db: tokio_rusqlite::Connection,
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
    let models = models::spawn_model_refresh(http_client.clone()).await;
    let db = db::init_db().await;
    let conversations: Arc<Mutex<HashMap<ChatId, Conversation>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let system_text0 = "You are a Telegram bot. In group chats you may see many messages, but only treat the latest message that explicitly mentions @<bot_name> (or replies to you) as the user's prompt; ignore the rest. Respond in plain text only (no Markdown).".to_string();
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

impl App {
    async fn process_message(&self, msg: Message) -> anyhow::Result<()> {
        if !is_common_text_message(&msg) {
            return Ok(());
        }

        self.maybe_update_user_name(&msg).await;

        let chat_id = msg.chat.id;

        let is_public = msg.chat.is_group() || msg.chat.is_supergroup() || msg.chat.is_channel();
        log::info!("received message from chat {}", chat_id);

        if is_public && !self.should_process_group_message(&msg).await {
            let user_message = self.extract_user_message(&msg).await?;
            self.persist_messages(chat_id, std::slice::from_ref(&user_message))
                .await;
            log::info!("ignored group message without mention for chat {}", chat_id);
            return Ok(());
        }

        if is_from_bot(&msg) {
            log::info!("ignoring message from bot account in chat {}", msg.chat.id);
            return Ok(());
        }

        self.ensure_authorized(chat_id).await?;

        let message_text = msg.text().unwrap().trim();
        if is_command(message_text) {
            if !is_public {
                self.process_command(chat_id, message_text).await?;
            }

            return Ok(());
        }

        let user_message = self.extract_user_message(&msg).await?;
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

        self.handle_llm_response(chat_id, msg.id, is_public, user_message, llm_response)
            .await
    }

    async fn ensure_authorized(&self, chat_id: ChatId) -> anyhow::Result<()> {
        if self.get_conversation(chat_id).await.is_authorized {
            return Ok(());
        }

        let message = format!(
            "You are not authorized to use this bot. Chat id {}",
            chat_id
        );
        self.bot.send_message(chat_id, &message).await?;
        Err(anyhow::anyhow!("Unauthorized"))
    }

    /// In group chats, only process messages that mention or reply to the bot; otherwise, just record them.
    async fn should_process_group_message(&self, msg: &Message) -> bool {
        let bot_username = self.bot_username.to_ascii_lowercase();

        let mentions_bot = msg
            .text()
            .map(|t| {
                t.to_ascii_lowercase()
                    .contains(&format!("@{}", bot_username))
            })
            .unwrap_or(false);

        let is_reply_to_bot = msg
            .reply_to_message()
            .and_then(|m| m.from.as_ref())
            .map(|user| {
                user.is_bot
                    && user
                        .username
                        .as_deref()
                        .map(|u| u.eq_ignore_ascii_case(&bot_username))
                        .unwrap_or(false)
            })
            .unwrap_or(false);

        mentions_bot || is_reply_to_bot
    }

    async fn handle_llm_response(
        &self,
        chat_id: ChatId,
        msg_id: MessageId,
        is_group: bool,
        user_message: conversation::Message,
        llm_response: anyhow::Result<openrouter_api::Response>,
    ) -> anyhow::Result<()> {
        match llm_response {
            Ok(llm_response) => {
                let reply_to = if is_group { Some(msg_id) } else { None };
                telegram::bot_split_send(
                    &self.bot,
                    chat_id,
                    &llm_response.completion_text,
                    reply_to,
                )
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
                    .set_message_reaction(chat_id, msg_id)
                    .reaction(vec![ReactionType::Emoji {
                        emoji: "🖕".to_string(),
                    }])
                    .await?;
            }
        }

        Ok(())
    }

    async fn maybe_update_user_name(&self, msg: &Message) {
        let user_name = if msg.chat.is_group() || msg.chat.is_supergroup() {
            msg.chat.title().map(str::to_owned)
        } else {
            let Some(user) = msg.from.as_ref() else {
                return;
            };

            user.username.clone().or_else(|| {
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
            })
        };

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

    async fn process_command(&self, chat_id: ChatId, message_text: &str) -> anyhow::Result<()> {
        let command = match commands::parse_command(message_text, &self.bot_username) {
            Ok(commands::Command::Ignore) => {
                // Command addressed to a different bot; ignore silently.
                return Ok(());
            }
            Ok(command) => command,
            Err(message) => {
                log::warn!("Failed to parse command: {}", message);
                self.bot.send_message(chat_id, message).await?;
                return Ok(());
            }
        };

        log::info!("Received command: {:?}", command);
        match command {
            commands::Command::Ignore => {
                // Command addressed to a different bot; ignore silently.
            }
            commands::Command::Help | commands::Command::Start => {
                let message = [
                    "Commands:",
                    "/help - show this help",
                    "/start - show this help",
                    "/models - list available models",
                    "/model [id|none] - show or set model",
                    "/key [key|none] - show or set API key",
                    "/system_prompt [text|none] - show or set system prompt",
                    "/approve [chat_id true|false] - admin only",
                ]
                .join("\n");
                telegram::bot_split_send(&self.bot, chat_id, &message, None).await?;
            }
            commands::Command::Models => {
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
                            Some(format!(
                                "`{}` \\- {}",
                                telegram::escape_markdown_v2(&f.id),
                                telegram::escape_markdown_v2(&f.name)
                            ))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let message = format!("Available models\\:\n{}", models);
                bot_split_send_formatted(&self.bot, chat_id, &message, None, ParseMode::MarkdownV2)
                    .await?;
            }
            commands::Command::Model(arg) => match arg {
                commands::CommandArg::Empty => {
                    let current_model_id = {
                        let conv = self.get_conversation(chat_id).await;
                        conv.model_id.clone()
                    };
                    let model = self.resolve_model(current_model_id.as_deref()).await;
                    self.bot
                        .send_message(
                            chat_id,
                            format!(
                                "Current model\\: `{}`",
                                telegram::escape_markdown_v2(&model.id)
                            ),
                        )
                        .parse_mode(ParseMode::MarkdownV2)
                        .await?;
                }
                commands::CommandArg::None => {
                    {
                        let mut conv = self.get_conversation(chat_id).await;
                        let old_model = self.resolve_model(conv.model_id.as_deref()).await;
                        conv.model_id = None;
                        let new_model = self.resolve_model(None).await;
                        let should_reload = old_model.id != new_model.id
                            && new_model.context_length >= old_model.context_length;
                        if should_reload {
                            db::load_history(&self.db, &mut conv, new_model.token_budget()).await;
                        }
                    }
                    db::set_model_id(&self.db, chat_id, None).await;
                    self.bot
                        .send_message(chat_id, "Model cleared; using default.")
                        .await?;
                }
                commands::CommandArg::Text(model_id) => {
                    let available_models = self.models.read().await;
                    let selected_model = available_models.iter().find(|m| m.id == model_id);

                    if let Some(model) = selected_model {
                        {
                            let mut conv = self.get_conversation(chat_id).await;
                            let old_model = self.resolve_model(conv.model_id.as_deref()).await;
                            conv.model_id = Some(model.id.clone());
                            let should_reload = old_model.id != model.id
                                && model.context_length >= old_model.context_length;
                            if should_reload {
                                db::load_history(&self.db, &mut conv, model.token_budget()).await;
                            }
                        }
                        db::set_model_id(&self.db, chat_id, Some(&model.id)).await;
                        log::info!("User {} selected model: `{}`", chat_id, model.name);
                        self.bot
                            .send_message(
                                chat_id,
                                format!(
                                    "Selected model\\: `{}`",
                                    telegram::escape_markdown_v2(&model.name)
                                ),
                            )
                            .parse_mode(ParseMode::MarkdownV2)
                            .await?;
                    } else {
                        log::warn!(
                            "User {} tried to select non-existent model: `{}`",
                            chat_id,
                            model_id
                        );
                        self.bot
                            .send_message(
                                chat_id,
                                format!(
                                    "Model not found\\: `{}`",
                                    telegram::escape_markdown_v2(&model_id)
                                ),
                            )
                            .parse_mode(ParseMode::MarkdownV2)
                            .await?;
                    }
                }
            },
            commands::Command::Key(arg) => match arg {
                commands::CommandArg::Empty => {
                    let current_key = {
                        let conv = self.get_conversation(chat_id).await;
                        conv.openrouter_api_key.clone()
                    };
                    match current_key {
                        Some(key) => {
                            let masked_key = mask_api_key(&key);
                            self.bot
                                .send_message(
                                    chat_id,
                                    format!(
                                        "API key is set \\(masked\\)\\: `{}`",
                                        telegram::escape_markdown_v2(&masked_key)
                                    ),
                                )
                                .parse_mode(ParseMode::MarkdownV2)
                                .await?;
                        }
                        None => {
                            self.bot.send_message(chat_id, "No API key set.").await?;
                        }
                    }
                }
                commands::CommandArg::None => {
                    {
                        let mut conv = self.get_conversation(chat_id).await;
                        conv.openrouter_api_key = None;
                    }
                    db::set_openrouter_api_key(&self.db, chat_id, None).await;
                    self.bot.send_message(chat_id, "API key cleared.").await?;
                }
                commands::CommandArg::Text(key) => {
                    {
                        let mut conv = self.get_conversation(chat_id).await;
                        conv.openrouter_api_key = Some(key.clone());
                    }
                    db::set_openrouter_api_key(&self.db, chat_id, Some(&key)).await;
                    self.bot.send_message(chat_id, "API key updated.").await?;
                }
            },
            commands::Command::SystemPrompt(arg) => match arg {
                commands::CommandArg::Empty => {
                    let current_prompt = {
                        let conv = self.get_conversation(chat_id).await;
                        conv.system_prompt.as_ref().map(|p| p.text.clone())
                    };
                    match current_prompt {
                        Some(prompt) => {
                            self.bot
                                .send_message(
                                    chat_id,
                                    format!(
                                        "Current system prompt\\: ```\n{}\n```",
                                        telegram::escape_markdown_v2(&prompt)
                                    ),
                                )
                                .parse_mode(ParseMode::MarkdownV2)
                                .await?;
                        }
                        None => {
                            self.bot
                                .send_message(chat_id, "No system prompt set.")
                                .await?;
                        }
                    }
                }
                commands::CommandArg::None => {
                    {
                        let mut conv = self.get_conversation(chat_id).await;
                        conv.system_prompt = None;
                    }
                    db::set_system_prompt(&self.db, chat_id, None).await;
                    self.bot
                        .send_message(chat_id, "System prompt cleared.")
                        .await?;
                }
                commands::CommandArg::Text(prompt) => {
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
            commands::Command::Approve(approve) => {
                let is_admin = { self.get_conversation(chat_id).await.is_admin };
                if !is_admin {
                    self.bot
                        .send_message(chat_id, "You are not authorized to use /approve.")
                        .await?;
                    return Ok(());
                }

                match approve {
                    commands::ApproveArg::Empty => {
                        let pending = db::list_unauthorized_chats(&self.db).await;
                        if pending.is_empty() {
                            self.bot.send_message(chat_id, "No pending users.").await?;
                            return Ok(());
                        }

                        let mut lines = Vec::with_capacity(pending.len());
                        for (pending_id, name) in pending {
                            let display_name = name.unwrap_or_else(|| "unknown".to_string());
                            let display_name = escape_markdown_v2(&display_name);
                            lines.push(format!("`{}` \\- {}", pending_id, display_name));
                        }

                        let message = format!("Pending users\\:\n{}", lines.join("\n"));
                        bot_split_send_formatted(
                            &self.bot,
                            chat_id,
                            &message,
                            None,
                            ParseMode::MarkdownV2,
                        )
                        .await?;
                    }
                    commands::ApproveArg::ApproveChat {
                        chat_id: target_chat_id,
                        is_authorized,
                    } => {
                        let target_id = ChatId(target_chat_id);
                        let result =
                            db::set_is_authorized(&self.db, target_id, is_authorized).await;
                        if result.is_err() {
                            self.bot
                                .send_message(chat_id, "Failed to authorize chat")
                                .await?;
                        } else {
                            {
                                let mut conv_map = self.conversations.lock().await;
                                if let Some(conv) = conv_map.get_mut(&target_id) {
                                    conv.is_authorized = is_authorized;
                                }
                            }

                            let message =
                                format!("Chat {} approved: {}", target_chat_id, is_authorized);
                            self.bot.send_message(chat_id, message).await?;
                        }
                    }
                    commands::ApproveArg::Invalid => {
                        self.bot
                            .send_message(chat_id, "Usage: /approve <chat_id> <true|false>")
                            .await?;
                    }
                }
            }
        }
        Ok(())
    }

    async fn extract_user_message(&self, msg: &Message) -> anyhow::Result<conversation::Message> {
        let mut user_text = msg
            .text()
            .expect("Only text messages are supported.")
            .to_owned();

        if !user_text.starts_with('/') {
            let replied_text = msg
                .reply_to_message()
                .and_then(|reply| reply.text())
                .map(|text| text.trim())
                .filter(|text| !text.is_empty());

            if let Some(replied_text) = replied_text {
                let replied_quoted = replied_text
                    .lines()
                    .map(|line| format!("> {}", line))
                    .collect::<Vec<_>>()
                    .join("\n");

                let selection = msg
                    .quote()
                    .map(|quote| quote.text.as_str())
                    .map(|text| text.trim())
                    .filter(|text| !text.is_empty())
                    .map(|text| {
                        text.lines()
                            .map(|line| format!("> {}", line))
                            .collect::<Vec<_>>()
                            .join("\n")
                    });

                let quoted = match selection {
                    Some(selection) => format!("{}\n\n\n{}", replied_quoted, selection),
                    None => replied_quoted,
                };

                user_text = format!("{}\n\n{}", quoted, user_text);
            }
        }

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
struct LlmRequestReady {
    payload: serde_json::Value,
    openrouter_api_key: String,
}

#[derive(Debug)]
enum LlmRequestError {
    NoApiKeyProvided,
}

type LlmRequestResult = Result<LlmRequestReady, LlmRequestError>;

/// Return a minimally identifying, masked version of an API key, e.g. `sk-or-v1-bab...68c`.
fn mask_api_key(key: &str) -> String {
    if key.len() <= 8 {
        // Very short keys: show first up to 3 chars and mask the rest.
        let prefix_len = key.len().min(3);
        return format!("{}***", &key[..prefix_len]);
    }

    let prefix_len = key.len().min(11);
    let suffix_len = key.len().saturating_sub(prefix_len).clamp(1, 3);

    let prefix = &key[..prefix_len];
    let suffix = &key[key.len().saturating_sub(suffix_len)..];

    format!("{prefix}...{suffix}")
}

fn is_from_bot(msg: &Message) -> bool {
    msg.from.as_ref().map(|u| u.is_bot).unwrap_or(false)
}

fn is_common_text_message(msg: &Message) -> bool {
    matches!(msg.kind, MessageKind::Common(..)) && msg.text().is_some()
}

fn is_command(message_text: &str) -> bool {
    message_text.starts_with('/')
}
