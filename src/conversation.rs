use log::warn;
use std::{collections::VecDeque, fmt::Display, sync::Arc};
use tiktoken_rs::{CoreBPE, get_bpe_from_model, o200k_base};

#[derive(Debug)]
pub struct Conversation {
    pub chat_id: u64,
    pub turns: VecDeque<ChatTurn>,
    pub prompt_tokens: usize,
    pub is_authorized: bool,
    pub openai_api_key: String,
    pub system_prompt: Option<Message>,
}

#[derive(Debug, Clone)]
pub struct ChatTurn {
    pub user: Message,
    pub assistant: Message,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub text: String,
    pub tokens: usize,
}

#[derive(Debug, Clone)]
#[repr(u8)]
pub enum MessageRole {
    System = 0,
    User = 1,
    Assistant = 2,
}

#[derive(Clone)]
pub struct TokenCounter {
    bpe: Arc<CoreBPE>,
}

impl Conversation {
    pub fn add_turn(&mut self, turn: ChatTurn) {
        self.prompt_tokens += turn.total_tokens();
        self.turns.push_back(turn);
    }

    pub fn prune_to_token_budget(&mut self, max_prompt_tokens: usize) {
        while self.prompt_tokens > max_prompt_tokens {
            if let Some(removed) = self.turns.pop_front() {
                self.prompt_tokens = self.prompt_tokens.saturating_sub(removed.total_tokens());
            }
        }
    }
}

impl ChatTurn {
    fn total_tokens(&self) -> usize {
        self.user.tokens + self.assistant.tokens
    }
}

impl Message {
    pub fn with_text_and_tokens(text: String, tokens: usize) -> Self {
        Self { text, tokens }
    }
    pub fn with_text(text: String, tokenizer: &TokenCounter) -> Self {
        let tokens = tokenizer.count_text(&text);
        Self::with_text_and_tokens(text, tokens)
    }
}

impl TokenCounter {
    pub fn new(model_name: &str) -> Self {
        let bpe = get_bpe_from_model(model_name)
            .or_else(|_| o200k_base())
            .expect("failed to load tokenizer vocabulary")
            .into();
        Self { bpe }
    }

    pub fn count_text(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len()
    }
}

impl Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::System => write!(f, "system"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
        }
    }
}
impl TryFrom<u8> for MessageRole {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(MessageRole::System),
            1 => Ok(MessageRole::User),
            2 => Ok(MessageRole::Assistant),
            _ => Err(()),
        }
    }
}
