#![allow(unused_imports)]

mod conversation;
mod db;
mod openai_api;
mod typing;

use anyhow::{Context, anyhow};
use conversation::{Conversation, MessageRole, TokenCounter};
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
        let user_message = conversation::Message {
            role: MessageRole::User,
            tokens: self.tokenizer.count_text(&user_text),
            text: user_text,
            ..Default::default()
        };

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

        if user_message.text.starts_with("/") || conversation.command.is_some() {
            self.handle_command(&conversation, &user_message).await?;
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
            Ok(llm_text) => {
                self.process_llm_response(chat_id, llm_text, user_message)
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

    async fn handle_command(
        &self,
        conversation: &conversation::Conversation,
        user_message: &conversation::Message,
    ) -> anyhow::Result<()> {
        match conversation.command {
            Some(conversation::Command::Token) => {
                let token = user_message.text.trim().to_string();
                db::update_token(conversation.chat_id, token).await?;
            }
            Some(conversation::Command::SystemMessage) => {
                let system_message = user_message.text.trim().to_string();
                db::update_system_message(conversation.chat_id, system_message).await?;
            }
            None => {
                if user_message.text.starts_with("/token") {
                    conversation.command = Some(conversation::Command::Token);
                    self.bot
                        .send_message(ChatId(conversation.chat_id), "Please enter your token:")
                        .await?;
                } else if user_message.text.starts_with("/system_message") {
                    conversation.command = Some(conversation::Command::SystemMessage);
                    if let Some(current_system_prompt) = conversation.system_prompt {
                        self.bot
                            .send_message(ChatId(conversation.chat_id), "Current system message:")
                            .await?;
                        self.bot
                            .send_message(ChatId(conversation.chat_id), current_system_prompt.text)
                            .await?;
                    }
                    self.bot
                        .send_message(
                            ChatId(conversation.chat_id),
                            "Please enter your new system message:",
                        )
                        .await?;
                }
            }
        }

        Ok(())
    }

    async fn process_llm_response(
        &self,
        chat_id: ChatId,
        llm_text: String,
        user_message: conversation::Message,
    ) -> anyhow::Result<()> {
        let postprocessed_result = self.postprocess(llm_text.clone());

        let (postprocessed_text, send_result) = match postprocessed_result {
            Ok(postprocessed_blocks) => {
                let postprocessed_text = postprocessed_blocks.join("\n");
                let send_result = self
                    .send_response_to_bot(chat_id, postprocessed_blocks)
                    .await;

                (postprocessed_text, send_result)
            }
            Err(error) => {
                let error = format!("Failed to postprocess LLM response: {:?}", error);
                log::error!("{}", error);

                ("".to_string(), Err(anyhow!(error)))
            }
        };

        let assistant_message = conversation::Message {
            role: MessageRole::Assistant,
            tokens: self.tokenizer.count_text(&postprocessed_text),
            text: postprocessed_text,
            raw_text: llm_text,
            send_failed: send_result.is_err(),
        };
        let messages = [user_message, assistant_message];
        self.get_conversation(chat_id)
            .await
            .add_messages(messages.iter().cloned());
        db::add_messages(&self.db, chat_id, messages.into_iter()).await;

        send_result
    }

    async fn send_response_to_bot(
        &self,
        chat_id: ChatId,
        blocks: Vec<String>,
    ) -> anyhow::Result<()> {
        for block in blocks {
            self.bot
                .send_message(chat_id, block)
                .parse_mode(ParseMode::MarkdownV2)
                .await?;
        }

        Ok(())
    }

    fn postprocess(&self, assistant_text: String) -> anyhow::Result<Vec<String>> {
        md2tgmdv2::Converter::default().go(&assistant_text)
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
