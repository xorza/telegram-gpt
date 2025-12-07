use std::{collections::VecDeque, sync::Arc};

use genai::chat::{ChatMessage, ChatRequest};
use log::warn;
use tiktoken_rs::{CoreBPE, get_bpe_from_model, o200k_base};

#[derive(Default)]
pub struct Conversation {
    next_turn_id: u64,
    turns: VecDeque<ChatTurn>,
    prompt_tokens: usize,
}

impl Conversation {
    pub fn record_user_message(&mut self, tokenizer: &TokenCounter, user_text: String) -> u64 {
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

    pub fn record_assistant_response(
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

    pub fn discard_turn(&mut self, turn_id: u64) {
        if let Some(index) = self.turns.iter().position(|turn| turn.id == turn_id) {
            if let Some(removed_turn) = self.turns.remove(index) {
                self.prompt_tokens = self
                    .prompt_tokens
                    .saturating_sub(removed_turn.total_tokens());
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    pub fn build_chat_request(&self, system_prompt: Option<&str>) -> ChatRequest {
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

    pub fn prompt_token_count(&self) -> usize {
        self.prompt_tokens
    }

    pub fn prune_to_token_budget(&mut self, max_prompt_tokens: usize, system_prompt_tokens: usize) {
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
                warn!(
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

#[derive(Clone)]
pub struct TokenCounter {
    bpe: Arc<CoreBPE>,
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
