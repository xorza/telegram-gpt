use log::warn;
use std::{collections::VecDeque, fmt::Display, sync::Arc};
use tiktoken_rs::{CoreBPE, get_bpe_from_model, o200k_base};

#[derive(Debug, Default)]
pub struct Conversation {
    next_turn_id: u64,
    pub turns: VecDeque<ChatTurn>,
    pub prompt_tokens: usize,
}

#[derive(Debug)]
pub struct ChatTurn {
    id: u64,
    pub user: TokenizedMessage,
    pub assistant: Option<TokenizedMessage>,
}

#[derive(Debug, Clone)]
pub struct TokenizedMessage {
    pub text: String,
    pub tokens: usize,
}

#[derive(Debug, Clone)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

#[derive(Clone)]
pub struct TokenCounter {
    bpe: Arc<CoreBPE>,
}

impl Conversation {
    pub fn record_user_message(&mut self, tokenizer: &TokenCounter, user_text: String) -> u64 {
        let turn_id = self.next_turn_id;
        self.next_turn_id += 1;
        let user = TokenizedMessage::new(user_text, tokenizer);
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
            let assistant = TokenizedMessage::new(answer, tokenizer);
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

    pub fn prompt_token_count(&self) -> usize {
        self.prompt_tokens
    }

    pub fn prune_to_token_budget(&mut self, max_prompt_tokens: usize) {
        while self.prompt_tokens > max_prompt_tokens {
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
                    self.prompt_tokens
                );
                break;
            }
        }
    }
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

impl TokenizedMessage {
    pub fn new(text: String, tokenizer: &TokenCounter) -> Self {
        let tokens = tokenizer.count_text(&text);
        Self { text, tokens }
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
            MessageRole::System => write!(f, "develope"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
        }
    }
}
