use log::warn;
use std::{collections::VecDeque, fmt::Display, sync::Arc};

use crate::openrouter_api;

#[derive(Debug)]
pub struct Conversation {
    pub chat_id: u64,
    pub history: VecDeque<Message>,
    pub is_authorized: bool,
    pub openrouter_api_key: Option<String>,
    pub system_prompt: Option<Message>,
}

#[derive(Debug, Clone, Default)]
pub struct Message {
    pub role: MessageRole,
    pub text: String,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum MessageRole {
    #[default]
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

    pub fn prune_to_token_budget(&mut self, token_budget: u64) {
        assert!(token_budget > 0, "Token budget must be greater than zero");

        let mut estimated_tokens =
            openrouter_api::estimate_tokens(self.history.iter().map(|m| m.text.as_str()));

        while estimated_tokens > token_budget {
            if self.history.pop_front().is_none() {
                break;
            }
            estimated_tokens =
                openrouter_api::estimate_tokens(self.history.iter().map(|m| m.text.as_str()));
        }
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
