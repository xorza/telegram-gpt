#![allow(unused_imports)]

mod conversation;
mod openai_api;
mod typing;

use conversation::{Conversation, TokenCounter};
use flexi_logger::{Cleanup, Criterion, Duplicate, FileSpec, Logger, Naming};
use openai_api::send_with_web_search;
use std::clone;
use std::fmt::Debug;
use std::{collections::HashMap, sync::Arc};
use teloxide::types::CopyTextButton;
use teloxide::{
    prelude::*,
    types::{ChatId, Message, ReactionType},
};
use tokio::sync::Mutex;
use typing::TypingIndicator;

type DynError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_MODEL: &str = "gpt-4.1";
const DEFAULT_MAX_PROMPT_TOKENS: usize = 120_000;

#[derive(Clone)]
struct App {
    bot: Bot,
    http_client: Arc<reqwest::Client>,
    model: String,
    tokenizer: Arc<TokenCounter>,
    system_prompt: Option<String>,
    system_prompt_tokens: usize,
    max_prompt_tokens: usize,
    conversations: Arc<Mutex<HashMap<ChatId, Conversation>>>,
}

#[tokio::main]
async fn main() -> Result<(), DynError> {
    let app = init()?;

    teloxide::repl(app.bot.clone(), move |bot: Bot, msg: Message| {
        let app = app.clone();

        async move {
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

            let typing_guard = TypingIndicator::new(bot.clone(), chat_id);

            let (turn_id, prompt_tokens, history_messages) = {
                let mut conv_map = app.conversations.lock().await;
                let conversation = conv_map
                    .entry(chat_id)
                    .or_insert_with(Conversation::default);
                let turn_id = conversation.record_user_message(&app.tokenizer, user_text.clone());
                conversation.prune_to_token_budget(app.max_prompt_tokens, app.system_prompt_tokens);
                let history = conversation.messages();
                let prompt_tokens = conversation.prompt_token_count() + app.system_prompt_tokens;

                (turn_id, prompt_tokens, history)
            };
            log::debug!(
                "chat {chat_id} prompt tokens: {prompt_tokens}/{max_prompt_tokens}",
                max_prompt_tokens = app.max_prompt_tokens
            );

            let llm_result = send_with_web_search(
                &app.http_client,
                &app.model,
                app.system_prompt.as_deref(),
                &history_messages,
            )
            .await;

            drop(typing_guard);

            let mut conv_map = app.conversations.lock().await;
            let conversation = conv_map
                .get_mut(&chat_id)
                .expect("conversation should exist");

            match llm_result {
                Ok(answer) => {
                    bot.send_message(chat_id, answer.clone()).await?;

                    conversation.record_assistant_response(&app.tokenizer, turn_id, answer);
                    conversation
                        .prune_to_token_budget(app.max_prompt_tokens, app.system_prompt_tokens);
                }
                Err(err) => {
                    log::error!("failed to get llm response: {err}");

                    conversation.discard_turn(turn_id);
                    if conversation.is_empty() {
                        conv_map.remove(&chat_id);
                    }

                    bot.set_message_reaction(chat_id, message_id)
                        .reaction(vec![ReactionType::Emoji {
                            emoji: "🔥".to_string(),
                        }])
                        .await?;
                }
            }

            respond(())
        }
    })
    .await;

    Ok(())
}

fn init() -> anyhow::Result<App, anyhow::Error> {
    dotenv::dotenv().ok();

    // Log to rotating files capped at 10MB each, keeping the 3 newest, while also duplicating info logs to stdout.
    Logger::try_with_env_or_str("info")?
        .log_to_file(FileSpec::default().directory("logs"))
        .rotate(
            Criterion::Size(10 * 1024 * 1024),
            Naming::Numbers,
            Cleanup::KeepLogFiles(3),
        )
        .duplicate_to_stdout(Duplicate::Warn)
        .start()?;

    log::info!("starting tggpt bot");

    let bot = Bot::from_env();
    let http_client = Arc::new(reqwest::Client::new());
    let model = std::env::var("GENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let tokenizer = Arc::new(TokenCounter::new(&model));
    let system_prompt = std::env::var("GENAI_SYSTEM_PROMPT").ok();
    let system_prompt_tokens = system_prompt
        .as_deref()
        .map(|prompt| tokenizer.count_text(prompt))
        .unwrap_or(0);
    let max_prompt_tokens = std::env::var("GENAI_MAX_PROMPT_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_MAX_PROMPT_TOKENS);
    let conversations: Arc<Mutex<HashMap<ChatId, Conversation>>> =
        Arc::new(Mutex::new(HashMap::new()));

    Ok(App {
        bot,
        http_client,
        model,
        tokenizer,
        system_prompt,
        system_prompt_tokens,
        max_prompt_tokens,
        conversations,
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
            .field("system_prompt_tokens", &self.system_prompt_tokens)
            .field("max_prompt_tokens", &self.max_prompt_tokens)
            .field("conversations", &self.conversations)
            .finish()
    }
}
