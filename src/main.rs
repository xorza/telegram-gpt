#![allow(unused_imports)]

mod conversation;
mod db;
mod openai_api;
mod typing;

use anyhow::{Context, anyhow};
use conversation::{Conversation, MessageRole, TokenCounter};
use db::init_db;
use db::load_conversation;
use flexi_logger::{Cleanup, Criterion, Duplicate, FileSpec, Logger, Naming};
use openai_api::{context_length, send_with_web_search};
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
    types::{ChatId, ReactionType},
};
use tokio::sync::{MappedMutexGuard, Mutex, MutexGuard};
use typing::TypingIndicator;

type DynError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_OPEN_AI_MODEL: &str = "gpt-4.1";

#[derive(Clone)]
struct App {
    bot: Bot,
    http_client: Arc<reqwest::Client>,
    model: String,
    tokenizer: Arc<TokenCounter>,
    system_prompt: Option<conversation::Message>,
    max_prompt_tokens: usize,
    conversations: Arc<Mutex<HashMap<ChatId, Conversation>>>,
    db: Arc<Mutex<Connection>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let app = init().await?;

    teloxide::repl(app.bot.clone(), move |bot: Bot, msg: Message| {
        let app = app.clone();
        async move {
            let result = process_message(app.clone(), bot, msg).await;
            if let Err(err) = result {
                log::error!("Error processing message: {}", err);
            }

            respond(())
        }
    })
    .await;

    Ok(())
}

async fn process_message(app: App, bot: Bot, msg: Message) -> anyhow::Result<()> {
    if msg.text().is_none() {
        bot.send_message(
            msg.chat.id,
            "Please send text messages so I can ask the language model.",
        )
        .await?;

        return Ok(());
    }

    let user_text = msg.text().unwrap().to_owned();
    let chat_id = msg.chat.id;
    let message_id = msg.id;

    log::info!("received message {message_id} from chat {chat_id}");

    let (turn_id, llm_result) = {
        let _typing_indicator = TypingIndicator::new(bot.clone(), chat_id);

        let mut conversation = app.get_conversation(chat_id).await?;
        if !conversation.is_authorized {
            let error = format!("Unauthorized user");
            log::warn!("{}", error);
            return Err(anyhow::anyhow!(error));
        }

        let message = conversation::Message::with_text(user_text, &app.tokenizer);

        let turn_id = conversation.record_user_message(message);

        let system_prompt_tokens = app.system_prompt.as_ref().map_or(0, |p| p.tokens);
        conversation.prune_to_token_budget(app.max_prompt_tokens - system_prompt_tokens);

        log::debug!(
            "chat {chat_id} prompt tokens: {prompt_tokens}/{max_prompt_tokens}",
            prompt_tokens = conversation.prompt_token_count() + system_prompt_tokens,
            max_prompt_tokens = app.max_prompt_tokens
        );

        let llm_result = send_with_web_search(
            &app.http_client,
            &app.model,
            app.system_prompt.as_ref(),
            &conversation,
        )
        .await;

        (turn_id, llm_result)
    };

    {
        let mut conversation = app.get_conversation(chat_id).await?;

        match llm_result {
            Ok(answer) => {
                let message = conversation::Message::with_text(answer.clone(), &app.tokenizer);
                conversation.record_assistant_response(turn_id, message);
                bot.send_message(chat_id, answer).await?;
            }
            Err(err) => {
                log::error!("failed to get llm response: {err}");

                conversation.discard_turn(turn_id);

                bot.set_message_reaction(chat_id, message_id)
                    .reaction(vec![ReactionType::Emoji {
                        emoji: "🖕".to_string(),
                    }])
                    .await?;
            }
        }
    }

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
        .duplicate_to_stdout(Duplicate::Debug)
        .start()?;

    let bot = Bot::from_env();
    let http_client = Arc::new(reqwest::Client::new());
    let model =
        std::env::var("OPEN_AI_MODEL").unwrap_or_else(|_| DEFAULT_OPEN_AI_MODEL.to_string());
    let tokenizer = Arc::new(TokenCounter::new(&model));
    let system_prompt = std::env::var("OPEN_AI_SYSTEM_PROMPT")
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|s| Some(conversation::Message::with_text(s, &tokenizer)));

    let max_prompt_tokens = context_length(&model);

    let db = Arc::new(Mutex::new(init_db()?));

    let conversations: Arc<Mutex<HashMap<ChatId, Conversation>>> =
        Arc::new(Mutex::new(HashMap::new()));

    log::info!("starting tggpt bot with model: {model}, max prompt tokens: {max_prompt_tokens}");

    Ok(App {
        bot,
        http_client,
        model,
        tokenizer,
        system_prompt,
        max_prompt_tokens,
        conversations,
        db,
    })
}

impl Debug for App {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("App")
            .field("bot", &self.bot)
            .field("http_client", &self.http_client)
            .field("model", &self.model)
            // .field("tokenizer", &self.tokenizer)
            .field("system_prompt", &self.system_prompt)
            .field("max_prompt_tokens", &self.max_prompt_tokens)
            .field("conversations", &self.conversations)
            .finish()
    }
}

impl App {
    async fn get_conversation(
        &self,
        chat_id: ChatId,
    ) -> anyhow::Result<MappedMutexGuard<'_, Conversation>> {
        let mut conv_map = self.conversations.lock().await;

        if !conv_map.contains_key(&chat_id) {
            // Blocking DB read; acceptable given small, infrequent loads.
            let conn = self.db.lock().await;
            let conv = load_conversation(chat_id, &conn)?;
            conv_map.insert(chat_id, conv);
        }

        Ok(MutexGuard::map(conv_map, |map| {
            map.get_mut(&chat_id).expect("conversation must exist")
        }))
    }
}
