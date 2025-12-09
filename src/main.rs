#![allow(unused_imports)]

mod conversation;
mod db;
mod openai_api;
mod typing;

use anyhow::{Context, anyhow};
use conversation::{Conversation, MessageRole, TokenCounter};
use diesel::dsl::Find;
use flexi_logger::{Cleanup, Criterion, Duplicate, FileSpec, Logger, Naming};
use log::error;
use reqwest::header::PROXY_AUTHENTICATE;
use rusqlite::Connection;
use serde_json::Value;
use std::fmt::Debug;
use std::process::Stdio;
use std::{collections::HashMap, path::Path, sync::Arc};
use teloxide::RequestError;
use teloxide::types::{CopyTextButton, ParseMode};
use teloxide::utils::markdown::blockquote;
use teloxide::{
    prelude::*,
    types::{ChatId, MessageId, ReactionType},
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::Command,
    sync::{MappedMutexGuard, Mutex, MutexGuard},
};
use typing::TypingIndicator;

const DEFAULT_OPEN_AI_MODEL: &str = "gpt-4.1";
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
    dotenv::dotenv().ok();

    verify_python3().await;

    // Log to rotating files capped at 10MB each, keeping the 3 newest, while also duplicating info logs to stdout.
    Logger::try_with_env_or_str("info")
        .expect("Failed to initialize logger")
        .log_to_file(FileSpec::default().directory("logs"))
        .rotate(
            Criterion::Size(10 * 1024 * 1024),
            Naming::Numbers,
            Cleanup::KeepLogFiles(10),
        )
        .duplicate_to_stdout(Duplicate::All)
        .start()
        .expect("Failed to start logger");

    let bot = Bot::from_env();
    let http_client = Arc::new(reqwest::Client::new());
    let model =
        std::env::var("OPEN_AI_MODEL").unwrap_or_else(|_| DEFAULT_OPEN_AI_MODEL.to_string());
    let tokenizer = Arc::new(TokenCounter::new(&model));

    let max_prompt_tokens = openai_api::context_length(&model);

    let db = Arc::new(Mutex::new(db::init_db()));

    let conversations: Arc<Mutex<HashMap<ChatId, Conversation>>> =
        Arc::new(Mutex::new(HashMap::new()));

    log::info!("starting tggpt bot with model: {model}, max prompt tokens: {max_prompt_tokens}");

    App {
        bot,
        http_client,
        model,
        tokenizer,
        max_prompt_tokens,
        conversations,
        db,
    }
}

async fn verify_python3() {
    // Ensure python3 and telegramify_markdown are available for markdown postprocessing.
    let probe = Command::new("python3")
        .args(["-c", "import telegramify_markdown; print('ok')"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .expect("failed to invoke python3 for dependency check");

    let probe_ok = probe.status.success() && String::from_utf8_lossy(&probe.stdout).trim() == "ok";

    if !probe_ok {
        let error = format!(
            "python3 with telegramify_markdown is required to run telegramify.py.\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&probe.stdout),
            String::from_utf8_lossy(&probe.stderr)
        );
        log::error!("{}", error);
        panic!("{}", error);
    }

    log::info!("python3 with telegramify_markdown is available");
}

impl App {
    async fn process_message(&self, msg: Message) -> anyhow::Result<()> {
        if msg.text().is_none() {
            self.bot
                .send_message(msg.chat.id, "Only text messages are supported.")
                .await?;

            return Ok(());
        }

        let chat_id = msg.chat.id;
        let user_text = msg.text().unwrap().to_owned();
        let user_message =
            conversation::Message::with_text(MessageRole::User, user_text, &self.tokenizer);

        let typing_indicator = TypingIndicator::new(self.bot.clone(), chat_id);

        log::info!("received message from chat {chat_id}");

        let mut conversation = self.get_conversation(chat_id).await;
        if !conversation.is_authorized {
            log::warn!("Unauthorized user {}", chat_id);

            let message = format!(
                "You are not authorized to use this bot. Chat id {}",
                chat_id
            );
            self.bot.send_message(msg.chat.id, &message).await?;
            return Ok(());
        }

        let system_prompt_tokens = conversation.system_prompt.as_ref().map_or(0, |p| p.tokens);
        conversation.prune_to_token_budget(
            self.max_prompt_tokens - system_prompt_tokens - user_message.tokens,
        );

        let history = conversation
            .system_prompt
            .as_ref()
            .into_iter()
            .chain(conversation.history.iter())
            .chain(std::iter::once(&user_message));

        let payload = openai_api::prepare_payload(&self.model, history, STREAM_RESPONSE);
        let openai_api_key = conversation.openai_api_key.clone();

        drop(conversation);

        let llm_result = openai_api::send(&self.http_client, &openai_api_key, payload).await;

        drop(typing_indicator);

        match llm_result {
            Ok(assistant_text) => {
                self.process_llm_response(chat_id, assistant_text, user_message)
                    .await?;
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

    async fn process_llm_response(
        &self,
        chat_id: ChatId,
        assistant_text: String,
        user_message: conversation::Message,
    ) -> anyhow::Result<()> {
        let blocks = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            self.postprocess(assistant_text.clone()),
        )
        .await
        .expect("postprocess timed out");

        let assistant_message = conversation::Message::with_text(
            MessageRole::Assistant,
            blocks.join("\n"),
            &self.tokenizer,
        );
        let messages = [user_message, assistant_message];
        self.get_conversation(chat_id)
            .await
            .add_messages(messages.iter().cloned());
        db::add_messages(&self.db, chat_id, messages.into_iter()).await;

        for block in blocks {
            self.bot
                .send_message(chat_id, block)
                .parse_mode(ParseMode::MarkdownV2)
                .await?;
        }

        Ok(())
    }

    async fn postprocess(&self, assistant_text: String) -> Vec<String> {
        let mut child = Command::new("python3")
            .arg("telegramify.py")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn telegramify.py");

        let mut stdin = child.stdin.take().expect("Failed to take stdin");
        stdin
            .write_all(assistant_text.as_bytes())
            .await
            .expect("failed to write to telegramify stdin");
        stdin
            .shutdown()
            .await
            .expect("failed to close telegramify stdin");
        drop(stdin);

        let mut stdout_buf = Vec::new();
        let mut stdout = child.stdout.take().expect("Failed to take stdout");
        let _ = stdout
            .read_to_end(&mut stdout_buf)
            .await
            .context("failed to read telegramify stdout");

        let mut stderr_buf = Vec::new();
        let mut stderr = child.stderr.take().expect("Failed to take stderr");
        let _ = stderr
            .read_to_end(&mut stderr_buf)
            .await
            .context("failed to read telegramify stderr");

        let status = child.wait().await.context("failed to wait for telegramify");
        if let Err(status) = status {
            let err_txt = String::from_utf8_lossy(&stderr_buf);
            let err_txt = format!("telegramify.py exited with {status}: {err_txt}");
            log::error!("{}", err_txt);
            panic!("{}", err_txt)
        }

        stdout_buf
            .split(|b| *b == 0)
            .filter(|b| !b.is_empty())
            .map(|block| String::from_utf8_lossy(block).to_string())
            .collect::<Vec<String>>()
    }

    async fn get_conversation(&self, chat_id: ChatId) -> MappedMutexGuard<'_, Conversation> {
        let mut conv_map = self.conversations.lock().await;

        if !conv_map.contains_key(&chat_id) {
            let conv =
                db::load_conversation(&self.db, chat_id, &self.tokenizer, self.max_prompt_tokens)
                    .await;
            conv_map.insert(chat_id, conv);
        }

        MutexGuard::map(conv_map, |map| {
            map.get_mut(&chat_id).expect("conversation must exist")
        })
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
