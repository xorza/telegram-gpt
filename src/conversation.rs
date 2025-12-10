use log::warn;
use std::{collections::VecDeque, fmt::Display, sync::Arc};
use tiktoken_rs::{CoreBPE, get_bpe_from_model, o200k_base};

#[derive(Debug)]
pub struct Conversation {
    pub chat_id: u64,
    pub history: VecDeque<Message>,
    pub prompt_tokens: usize,
    pub is_authorized: bool,
    pub openai_api_key: String,
    pub system_prompt: Option<Message>,
}

#[derive(Debug, Clone, Default)]
pub struct Message {
    pub role: MessageRole,
    pub tokens: usize,
    pub text: String,
    pub raw_text: String,
    pub send_failed: bool,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
    pub fn add_message(&mut self, message: Message) {
        // Token count is managed by callers during reconstruction/loading.
        self.history.push_back(message);
    }

    pub fn add_messages<I>(&mut self, messages: I)
    where
        I: IntoIterator<Item = Message>,
    {
        for mut message in messages {
            self.prompt_tokens += message.tokens;
            // dont need raw_text in history
            message.raw_text = String::new();
            self.history.push_back(message);
        }
    }

    pub fn prune_to_token_budget(&mut self, max_prompt_tokens: usize) {
        while self.prompt_tokens > max_prompt_tokens {
            if let Some(removed) = self.history.pop_front() {
                self.prompt_tokens = self.prompt_tokens.saturating_sub(removed.tokens);
            }
        }
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

impl Default for MessageRole {
    fn default() -> Self {
        MessageRole::System
    }
}
