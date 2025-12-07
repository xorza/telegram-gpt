use flexi_logger::{Cleanup, Criterion, Duplicate, FileSpec, Logger, Naming};
use genai::{
    chat::{ChatMessage, ChatRequest},
    Client,
};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};
use teloxide::{
    prelude::*,
    types::{ChatId, Message},
};
use tokio::sync::Mutex;

type DynError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_MODEL: &str = "gpt-4.1";
const MAX_TURNS_PER_CHAT: usize = 200;

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
    let client = Client::default();
    let model = std::env::var("GENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let system_prompt = std::env::var("GENAI_SYSTEM_PROMPT").ok();
    let conversations: Arc<Mutex<HashMap<ChatId, Conversation>>> =
        Arc::new(Mutex::new(HashMap::new()));

    teloxide::repl(bot, move |bot: Bot, msg: Message| {
        let client = client.clone();
        let model = model.clone();
        let system_prompt = system_prompt.clone();
        let conversations = conversations.clone();
        async move {
            if let Some(text) = msg.text() {
                let user_text = text.to_owned();
                let chat_id = msg.chat.id;
                let (chat_request, turn_id) = {
                    let mut conv_map = conversations.lock().await;
                    let conversation = conv_map
                        .entry(chat_id)
                        .or_insert_with(Conversation::default);
                    let turn_id = conversation.record_user_message(user_text.clone());
                    let chat_request = conversation.build_chat_request(system_prompt.as_deref());
                    (chat_request, turn_id)
                };

                match send_to_llm(&client, &model, chat_request).await {
                    Ok(answer) => {
                        bot.send_message(chat_id, answer.clone()).await?;
                        let mut conv_map = conversations.lock().await;
                        let conversation = conv_map
                            .get_mut(&chat_id)
                            .expect("conversation should exist");
                        conversation.record_assistant_response(turn_id, answer);
                        conversation.trim_history();
                    }
                    Err(err) => {
                        log::error!("failed to get llm response: {err}");
                        bot.send_message(
                            chat_id,
                            "I couldn't reach the language model. Please try again.",
                        )
                        .await?;
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

async fn send_to_llm(
    client: &Client,
    model: &str,
    chat_request: ChatRequest,
) -> Result<String, DynError> {
    let chat_res = client.exec_chat(model, chat_request, None).await?;

    let answer = chat_res
        .first_text()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "The language model returned an empty response.".to_string());

    Ok(answer)
}

#[derive(Default)]
struct Conversation {
    next_turn_id: u64,
    turns: VecDeque<ChatTurn>,
}

impl Conversation {
    fn record_user_message(&mut self, user_text: String) -> u64 {
        let turn_id = self.next_turn_id;
        self.next_turn_id += 1;
        self.turns.push_back(ChatTurn {
            id: turn_id,
            user: ChatMessage::user(user_text),
            assistant: None,
        });
        turn_id
    }

    fn record_assistant_response(&mut self, turn_id: u64, answer: String) {
        if let Some(turn) = self.turns.iter_mut().find(|turn| turn.id == turn_id) {
            turn.assistant = Some(ChatMessage::assistant(answer));
        }
    }

    fn discard_turn(&mut self, turn_id: u64) {
        if let Some(index) = self.turns.iter().position(|turn| turn.id == turn_id) {
            self.turns.remove(index);
        }
    }

    fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    fn build_chat_request(&self, system_prompt: Option<&str>) -> ChatRequest {
        let mut messages = Vec::with_capacity(self.turns.len() * 2 + 1);
        for turn in &self.turns {
            messages.push(turn.user.clone());
            if let Some(assistant) = &turn.assistant {
                messages.push(assistant.clone());
            }
        }

        let chat_req = ChatRequest::new(messages);
        if let Some(prompt) = system_prompt {
            chat_req.with_system(prompt.to_owned())
        } else {
            chat_req
        }
    }

    fn trim_history(&mut self) {
        while self.turns.len() > MAX_TURNS_PER_CHAT {
            if self
                .turns
                .front()
                .map(|turn| turn.assistant.is_some())
                .unwrap_or(false)
            {
                self.turns.pop_front();
            } else {
                break;
            }
        }
    }
}

struct ChatTurn {
    id: u64,
    user: ChatMessage,
    assistant: Option<ChatMessage>,
}
