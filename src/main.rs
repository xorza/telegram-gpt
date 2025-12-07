use flexi_logger::{Cleanup, Criterion, Duplicate, FileSpec, Logger, Naming};
use genai::{
    Client,
    chat::{ChatMessage, ChatRequest},
};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};
use teloxide::{
    prelude::*,
    types::{ChatId, Message},
};
use tiktoken_rs::{CoreBPE, get_bpe_from_model, o200k_base};
use tokio::sync::Mutex;

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
    let client = Client::default();
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
        let client = client.clone();
        let model = model.clone();
        let system_prompt = system_prompt.clone();
        let tokenizer = tokenizer.clone();
        let system_prompt_tokens = system_prompt_tokens;
        let max_prompt_tokens = max_prompt_tokens;
        let conversations = conversations.clone();
        async move {
            if let Some(text) = msg.text() {
                let user_text = text.to_owned();
                let chat_id = msg.chat.id;
                let (chat_request, turn_id, prompt_tokens) = {
                    let mut conv_map = conversations.lock().await;
                    let conversation = conv_map
                        .entry(chat_id)
                        .or_insert_with(Conversation::default);
                    let turn_id = conversation.record_user_message(&tokenizer, user_text.clone());
                    conversation.prune_to_token_budget(max_prompt_tokens, system_prompt_tokens);
                    let chat_request = conversation.build_chat_request(system_prompt.as_deref());
                    let prompt_tokens = conversation.prompt_token_count() + system_prompt_tokens;
                    (chat_request, turn_id, prompt_tokens)
                };
                log::debug!("chat {chat_id} prompt tokens: {prompt_tokens}/{max_prompt_tokens}");

                match send_to_llm(&client, &model, chat_request).await {
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
    prompt_tokens: usize,
}

impl Conversation {
    fn record_user_message(&mut self, tokenizer: &TokenCounter, user_text: String) -> u64 {
        let turn_id = self.next_turn_id;
        self.next_turn_id += 1;
        let user = TokenizedMessage::new_user(user_text, tokenizer);
        self.prompt_tokens += user.tokens;
        self.turns.push_back(ChatTurn {
            id: turn_id,
            user,
            assistant: None,
        });
        turn_id
    }

    fn record_assistant_response(
        &mut self,
        tokenizer: &TokenCounter,
        turn_id: u64,
        answer: String,
    ) {
        if let Some(turn) = self.turns.iter_mut().find(|turn| turn.id == turn_id) {
            let assistant = TokenizedMessage::new_assistant(answer, tokenizer);
            self.prompt_tokens += assistant.tokens;
            turn.assistant = Some(assistant);
        }
    }

    fn discard_turn(&mut self, turn_id: u64) {
        if let Some(index) = self.turns.iter().position(|turn| turn.id == turn_id) {
            if let Some(removed_turn) = self.turns.remove(index) {
                self.prompt_tokens = self
                    .prompt_tokens
                    .saturating_sub(removed_turn.total_tokens());
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    fn build_chat_request(&self, system_prompt: Option<&str>) -> ChatRequest {
        let mut messages = Vec::with_capacity(self.turns.len() * 2 + 1);
        for turn in &self.turns {
            messages.push(turn.user.message.clone());
            if let Some(assistant) = &turn.assistant {
                messages.push(assistant.message.clone());
            }
        }

        let chat_req = ChatRequest::new(messages);
        if let Some(prompt) = system_prompt {
            chat_req.with_system(prompt.to_owned())
        } else {
            chat_req
        }
    }

    fn prompt_token_count(&self) -> usize {
        self.prompt_tokens
    }

    fn prune_to_token_budget(&mut self, max_prompt_tokens: usize, system_prompt_tokens: usize) {
        while self.prompt_tokens + system_prompt_tokens > max_prompt_tokens {
            if self
                .turns
                .front()
                .map(|turn| turn.is_complete())
                .unwrap_or(false)
            {
                if let Some(removed) = self.turns.pop_front() {
                    self.prompt_tokens = self.prompt_tokens.saturating_sub(removed.total_tokens());
                }
            } else {
                log::warn!(
                    "token budget exceeded but no complete turns left to drop (need <= {max_prompt_tokens}, have {})",
                    self.prompt_tokens + system_prompt_tokens
                );
                break;
            }
        }
    }
}

struct ChatTurn {
    id: u64,
    user: TokenizedMessage,
    assistant: Option<TokenizedMessage>,
}

impl ChatTurn {
    fn total_tokens(&self) -> usize {
        let assistant_tokens = self.assistant.as_ref().map(|msg| msg.tokens).unwrap_or(0);
        self.user.tokens + assistant_tokens
    }

    fn is_complete(&self) -> bool {
        self.assistant.is_some()
    }
}

#[derive(Clone)]
struct TokenizedMessage {
    message: ChatMessage,
    tokens: usize,
}

impl TokenizedMessage {
    fn new_user(text: String, tokenizer: &TokenCounter) -> Self {
        let tokens = tokenizer.count_text(&text);
        Self {
            message: ChatMessage::user(text),
            tokens,
        }
    }

    fn new_assistant(text: String, tokenizer: &TokenCounter) -> Self {
        let tokens = tokenizer.count_text(&text);
        Self {
            message: ChatMessage::assistant(text),
            tokens,
        }
    }
}

struct TokenCounter {
    bpe: Arc<CoreBPE>,
}

impl TokenCounter {
    fn new(model_name: &str) -> Self {
        let bpe = get_bpe_from_model(model_name)
            .or_else(|_| o200k_base())
            .expect("failed to load tokenizer vocabulary")
            .into();
        Self { bpe }
    }

    fn count_text(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len()
    }
}
