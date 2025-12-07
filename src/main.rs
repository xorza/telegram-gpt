mod conversation;
mod openai_api;
mod typing;

use conversation::{Conversation, TokenCounter};
use flexi_logger::{Cleanup, Criterion, Duplicate, FileSpec, Logger, Naming};
use openai_api::send_with_web_search;
use std::{collections::HashMap, sync::Arc};
use teloxide::{
    prelude::*,
    types::{ChatId, Message, ReactionType},
};
use tokio::sync::Mutex;
use typing::TypingIndicator;

type DynError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_MODEL: &str = "gpt-4.1";
const DEFAULT_MAX_PROMPT_TOKENS: usize = 120_000;

#[tokio::main]
async fn main() -> Result<(), DynError> {
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

    teloxide::repl(bot, move |bot: Bot, msg: Message| {
        let model = model.clone();
        let system_prompt = system_prompt.clone();
        let tokenizer = tokenizer.clone();
        let system_prompt_tokens = system_prompt_tokens;
        let max_prompt_tokens = max_prompt_tokens;
        let conversations = conversations.clone();
        let http_client = http_client.clone();
        async move {
            if let Some(text) = msg.text() {
                let user_text = text.to_owned();
                let chat_id = msg.chat.id;
                let message_id = msg.id;
                let typing_guard = TypingIndicator::new(bot.clone(), chat_id);
                let (turn_id, prompt_tokens, history_messages) = {
                    let mut conv_map = conversations.lock().await;
                    let conversation = conv_map
                        .entry(chat_id)
                        .or_insert_with(Conversation::default);
                    let turn_id = conversation.record_user_message(&tokenizer, user_text.clone());
                    conversation.prune_to_token_budget(max_prompt_tokens, system_prompt_tokens);
                    let history = conversation.messages();
                    let prompt_tokens = conversation.prompt_token_count() + system_prompt_tokens;
                    (turn_id, prompt_tokens, history)
                };
                log::debug!("chat {chat_id} prompt tokens: {prompt_tokens}/{max_prompt_tokens}");

                let llm_result = send_with_web_search(
                    &http_client,
                    &model,
                    system_prompt.as_deref(),
                    &history_messages,
                )
                .await;
                drop(typing_guard);

                match llm_result {
                    Ok(answer) => {
                        bot.send_message(chat_id, answer.clone()).await?;
                        let mut conv_map = conversations.lock().await;
                        let conversation = conv_map
                            .get_mut(&chat_id)
                            .expect("conversation should exist");
                        conversation.record_assistant_response(&tokenizer, turn_id, answer);
                        conversation.prune_to_token_budget(max_prompt_tokens, system_prompt_tokens);
                    }
                    Err(err) => {
                        log::error!("failed to get llm response: {err}");

                        if let Err(reaction_err) = bot
                            .set_message_reaction(chat_id, message_id)
                            .reaction(vec![ReactionType::Emoji {
                                emoji: "🖕".to_string(),
                            }])
                            .await
                        {
                            log::warn!(
                                "failed to set failure reaction for chat {chat_id}: {reaction_err}"
                            );
                        }
                        let mut conv_map = conversations.lock().await;
                        let conversation = conv_map
                            .get_mut(&chat_id)
                            .expect("conversation should exist");
                        conversation.discard_turn(turn_id);
                        if conversation.is_empty() {
                            conv_map.remove(&chat_id);
                        }
                    }
                }
            } else {
                bot.send_message(
                    msg.chat.id,
                    "Please send text messages so I can ask the language model.",
                )
                .await?;
            }
            respond(())
        }
    })
    .await;

    Ok(())
}
