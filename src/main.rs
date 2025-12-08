#![allow(unused_imports)]

mod conversation;
mod openai_api;
mod typing;

use conversation::{Conversation, TokenCounter, TokenizedMessage};
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
use tokio::sync::{MappedMutexGuard, Mutex, MutexGuard};
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
    system_prompt: Option<TokenizedMessage>,
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
                let mut conversation = app.get_conversation(chat_id).await;
                let turn_id = conversation.record_user_message(&app.tokenizer, user_text.clone());
                let system_prompt_tokens = app.system_prompt.as_ref().map_or(0, |p| p.tokens);
                conversation.prune_to_token_budget(app.max_prompt_tokens, system_prompt_tokens);
                let history = conversation.messages();
                let prompt_tokens = conversation.prompt_token_count() + system_prompt_tokens;

                (turn_id, prompt_tokens, history)
            };
            log::debug!(
                "chat {chat_id} prompt tokens: {prompt_tokens}/{max_prompt_tokens}",
                max_prompt_tokens = app.max_prompt_tokens
            );

            let llm_result = send_with_web_search(
                &app.http_client,
                &app.model,
                app.system_prompt.as_ref(),
                &history_messages,
            )
            .await;

            drop(typing_guard);

            let mut conversation = app.get_conversation(chat_id).await;

            match llm_result {
                Ok(answer) => {
                    conversation.record_assistant_response(&app.tokenizer, turn_id, answer.clone());
                    bot.send_message(chat_id, answer).await?;
                }
                Err(err) => {
                    log::error!("failed to get llm response: {err}");

                    conversation.discard_turn(turn_id);

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

    let bot = Bot::from_env();
    let http_client = Arc::new(reqwest::Client::new());
    let model = std::env::var("GENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let tokenizer = Arc::new(TokenCounter::new(&model));
    let system_prompt = std::env::var("GENAI_SYSTEM_PROMPT")
        .ok()
        .filter(|s| !s.is_empty())
        .and_then(|s| Some(TokenizedMessage::new(s, &tokenizer)));

    let conversations: Arc<Mutex<HashMap<ChatId, Conversation>>> =
        Arc::new(Mutex::new(HashMap::new()));

    log::info!("starting tggpt bot");

    Ok(App {
        bot,
        http_client,
        model,
        tokenizer,
        system_prompt,
        max_prompt_tokens: DEFAULT_MAX_PROMPT_TOKENS,
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
            .field("max_prompt_tokens", &self.max_prompt_tokens)
            .field("conversations", &self.conversations)
            .finish()
    }
}

impl App {
    async fn get_conversation(&self, chat_id: ChatId) -> MappedMutexGuard<'_, Conversation> {
        // Lock the shared map then map the guard to just the Conversation for this chat_id
        let conv_map = self.conversations.lock().await;

        MutexGuard::map(conv_map, |map| {
            map.entry(chat_id).or_insert_with(Conversation::default)
        })
    }
}
