use crate::panic_handler::fatal_panic;
use teloxide::{
    payloads::SendMessageSetters,
    prelude::{Bot, Requester},
    types::{ChatId, MessageId, ParseMode, ReplyParameters},
};

const TELEGRAM_MAX_MESSAGE_LENGTH: usize = 4096;

/// Escape a string so it is safe to send with `ParseMode::MarkdownV2`.
pub fn escape_markdown_v2(text: &str) -> String {
    teloxide::utils::markdown::escape(text)
}

async fn send_formatted_checked(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    reply_to: Option<MessageId>,
    parse_mode: ParseMode,
) -> anyhow::Result<()> {
    assert!(
        text.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH,
        "message exceeds telegram max length"
    );

    match reply_to {
        Some(reply_id) => {
            let reply = ReplyParameters {
                message_id: reply_id,
                ..Default::default()
            };
            bot.send_message(chat_id, text)
                .reply_parameters(reply)
                .parse_mode(parse_mode.clone())
                .await?;
        }
        None => {
            bot.send_message(chat_id, text)
                .parse_mode(parse_mode.clone())
                .await?;
        }
    }

    Ok(())
}

pub async fn send_message_checked(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    reply_to: Option<MessageId>,
) -> anyhow::Result<()> {
    assert!(
        text.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH,
        "message exceeds telegram max length"
    );

    match reply_to {
        Some(reply_id) => {
            let reply = ReplyParameters {
                message_id: reply_id,
                ..Default::default()
            };
            bot.send_message(chat_id, text)
                .reply_parameters(reply)
                .await?;
        }
        None => {
            bot.send_message(chat_id, text).await?;
        }
    }

    Ok(())
}

/// Send a formatted message (e.g., MarkdownV2), splitting only on newlines.
/// Calls `fatal_panic` if any single line exceeds Telegram's maximum length.
pub async fn bot_split_send_formatted(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    reply_to: Option<MessageId>,
    parse_mode: ParseMode,
) -> anyhow::Result<()> {
    if text.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH {
        return send_formatted_checked(bot, chat_id, text, reply_to, parse_mode).await;
    }

    let mut buffer = String::new();
    let mut buffer_len = 0usize;

    for line in text.split('\n') {
        let line_len = line.chars().count();
        let required = line_len + if buffer.is_empty() { 0 } else { 1 }; // include newline

        if line_len > TELEGRAM_MAX_MESSAGE_LENGTH {
            fatal_panic("Formatted message contains a line longer than Telegram allows");
        }

        if buffer_len + required > TELEGRAM_MAX_MESSAGE_LENGTH {
            send_formatted_checked(bot, chat_id, &buffer, reply_to, parse_mode.clone()).await?;
            buffer.clear();
            buffer_len = 0;
        }

        if !buffer.is_empty() {
            buffer.push('\n');
            buffer_len += 1;
        }

        buffer.push_str(line);
        buffer_len += line_len;
    }

    if !buffer.is_empty() {
        send_formatted_checked(bot, chat_id, &buffer, reply_to, parse_mode).await?;
    }

    Ok(())
}

pub async fn bot_split_send(
    bot: &Bot,
    chat_id: ChatId,
    text: &str,
    reply_to: Option<MessageId>,
) -> anyhow::Result<()> {
    if text.chars().count() <= TELEGRAM_MAX_MESSAGE_LENGTH {
        send_message_checked(bot, chat_id, text, reply_to).await?;
        return Ok(());
    }

    let mut buffer = String::new();
    let mut buffer_len = 0usize;
    let mut chunk = String::new();
    let mut chunk_len = 0usize;

    for token in text.split_inclusive([' ', '\n']) {
        let token_len = token.chars().count();
        if token_len > TELEGRAM_MAX_MESSAGE_LENGTH {
            if !buffer.is_empty() {
                send_message_checked(bot, chat_id, &buffer, reply_to).await?;
                buffer.clear();
                buffer_len = 0;
            }
            for ch in token.chars() {
                if chunk_len == TELEGRAM_MAX_MESSAGE_LENGTH {
                    send_message_checked(bot, chat_id, &chunk, reply_to).await?;
                    chunk.clear();
                    chunk_len = 0;
                }
                chunk.push(ch);
                chunk_len += 1;
            }
            if !chunk.is_empty() {
                send_message_checked(bot, chat_id, &chunk, reply_to).await?;
                chunk.clear();
                chunk_len = 0;
            }
            continue;
        }
        if buffer_len + token_len > TELEGRAM_MAX_MESSAGE_LENGTH && !buffer.is_empty() {
            send_message_checked(bot, chat_id, &buffer, reply_to).await?;
            buffer.clear();
            buffer_len = 0;
        }

        buffer.push_str(token);
        buffer_len += token_len;
    }

    if !buffer.is_empty() {
        send_message_checked(bot, chat_id, &buffer, reply_to).await?;
    }

    Ok(())
}
