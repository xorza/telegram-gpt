use log::warn;
use std::{collections::VecDeque, fmt::Display, sync::Arc};

#[derive(Debug)]
pub struct Conversation {
    pub chat_id: u64,
    pub history: VecDeque<Message>,
    pub is_authorized: bool,
    pub openai_api_key: String,
    pub system_prompt: Option<Message>,
}

#[derive(Debug, Clone, Default)]
pub struct Message {
    pub role: MessageRole,
    pub text: String,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MessageRole {
    System = 0,
    User = 1,
    Assistant = 2,
}

impl Conversation {
    pub fn add_message(&mut self, message: Message) {
        self.history.push_back(message);
    }

    pub fn add_messages<I>(&mut self, messages: I)
    where
        I: IntoIterator<Item = Message>,
    {
        for message in messages {
            self.history.push_back(message);
        }
    }

    pub fn prune_to_token_budget(&mut self, context_length: u64) {
        unimplemented!()
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
