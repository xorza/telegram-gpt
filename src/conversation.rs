use log::warn;
use std::{collections::VecDeque, fmt::Display, sync::Arc};

#[derive(Debug)]
pub struct Conversation {
    pub chat_id: u64,
    pub history: VecDeque<Message>,
    pub prompt_tokens: u64,
    pub is_authorized: bool,
    pub openai_api_key: String,
    pub developer_prompt: Option<Message>,
}

#[derive(Debug, Clone, Default)]
pub struct Message {
    pub role: MessageRole,
    pub tokens: u64,
    pub text: String,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MessageRole {
    Developer = 0,
    User = 1,
    Assistant = 2,
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
        for message in messages {
            self.prompt_tokens += message.tokens;
            self.history.push_back(message);
        }
    }

    pub fn prune_to_token_budget(&mut self, max_prompt_tokens: u64) {
        while self.prompt_tokens > max_prompt_tokens {
            if let Some(removed) = self.history.pop_front() {
                self.prompt_tokens = self.prompt_tokens.saturating_sub(removed.tokens);
            }
        }
    }
}

impl Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::Developer => write!(f, "developer"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
        }
    }
}

impl TryFrom<u8> for MessageRole {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(MessageRole::Developer),
            1 => Ok(MessageRole::User),
            2 => Ok(MessageRole::Assistant),
            _ => Err(()),
        }
    }
}

impl Default for MessageRole {
    fn default() -> Self {
        MessageRole::Developer
    }
}
