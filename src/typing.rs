use teloxide::{prelude::*, types::ChatAction};
use tokio::{
    task::JoinHandle,
    time::{Duration, sleep},
};

pub struct TypingIndicator {
    handle: JoinHandle<()>,
}

impl TypingIndicator {
    pub fn new(bot: Bot, chat_id: ChatId) -> Self {
        let handle = tokio::spawn(async move {
            loop {
                if bot
                    .send_chat_action(chat_id, ChatAction::Typing)
                    .await
                    .is_err()
                {
                    break;
                }
                sleep(Duration::from_secs(4)).await;
            }
        });
        Self { handle }
    }
}

impl Drop for TypingIndicator {
    fn drop(&mut self) {
        self.handle.abort();
    }
}
