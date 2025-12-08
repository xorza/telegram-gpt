use crate::conversation::{self, ChatTurn, Conversation, MessageRole, TokenCounter};
use anyhow::Result;
use rusqlite::{Connection, Error as SqliteError};
use std::sync::Arc;
use teloxide::types::ChatId;
use tokio::sync::Mutex;

const SCHEMA_VERSION: i32 = 1;

pub fn init_db() -> Result<Connection> {
    let db_path = std::env::var("SQLITE_PATH").unwrap_or_else(|_| "data/db.sqlite".to_string());

    // Ensure parent directory exists
    if let Some(parent) = std::path::Path::new(&db_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let conn = Connection::open(&db_path)?;

    // Enable WAL for better concurrency; ignore errors silently.
    // let _ = conn.pragma_update(None, "journal_mode", "WAL");

    match std::env::var("DB_ENCRYPTION_KEY") {
        Ok(key) if !key.is_empty() => conn.pragma_update(None, "key", &key)?,
        _ => log::warn!("DB_ENCRYPTION_KEY not set; database will be unencrypted"),
    }

    // Initialize database schema if needed and validate version.
    let version = get_schema_version(&conn)?;
    if version == 0 {
        init_schema(&conn)?;
        set_schema_version(&conn, SCHEMA_VERSION)?;
        log::info!("Initialized database schema version {}", SCHEMA_VERSION);
    } else if version != SCHEMA_VERSION {
        panic!(
            "Unsupported database schema version {} (expected {})",
            version, SCHEMA_VERSION
        );
    } else {
        log::info!("Database schema version {} detected", version);
    }

    Ok(conn)
}

fn init_schema(conn: &Connection) -> Result<(), SqliteError> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            tokens  INTEGER NOT NULL,
            role    INTEGER NOT NULL,
            text    TEXT NOT NULL
        )",
        [],
    )?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS chats (
            chat_id INTEGER PRIMARY KEY NOT NULL,
            is_authorized BOOLEAN NOT NULL,
            open_ai_api_key  TEXT NOT NULL,
            system_prompt    TEXT NOT NULL
        )",
        [],
    )?;

    Ok(())
}

fn get_schema_version(conn: &Connection) -> Result<i32, SqliteError> {
    conn.query_row("PRAGMA user_version", [], |row| row.get(0))
}

fn set_schema_version(conn: &Connection, version: i32) -> Result<(), SqliteError> {
    conn.pragma_update(None, "user_version", version)
}

pub async fn load_conversation(
    db: &Arc<Mutex<Connection>>,
    chat_id: ChatId,
    tokenizer: &TokenCounter,
    max_tokens: usize,
) -> anyhow::Result<Conversation> {
    let (mut conversation, history) = {
        let conn = db.lock().await;

        let (is_authorized, open_ai_api_key, system_prompt) = {
            // Fetch exactly one chat row; panic if multiple rows are found.
            let mut stmt = conn.prepare(
                "SELECT is_authorized, open_ai_api_key, system_prompt \
            FROM chats WHERE chat_id = ?1 LIMIT 2",
            )?;
            let mut rows = stmt.query([chat_id.0])?;

            let (is_authorized, open_ai_api_key, system_prompt) = match rows.next()? {
                Some(row) => {
                    let is_authorized: bool = row.get(0)?;
                    let open_ai_api_key: String = row.get(1)?;
                    let system_prompt: String = row.get(2)?;
                    (is_authorized, open_ai_api_key, system_prompt)
                }
                None => {
                    let r = conn.execute(
                    "INSERT INTO chats (chat_id, is_authorized, open_ai_api_key, system_prompt) \
                     VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![chat_id.0, false, "", ""],
                )?;
                    if r != 1 {
                        let error = format!("failed to insert chat row for chat_id {}", chat_id.0);
                        log::error!("{}", error);
                        panic!("{}", error);
                    }

                    (false, String::new(), String::new())
                }
            };

            if rows.next()?.is_some() {
                panic!("multiple chat rows found for chat_id {}", chat_id.0);
            }
            (is_authorized, open_ai_api_key, system_prompt)
        };

        let system_prompt = if !system_prompt.is_empty() {
            Some(conversation::Message::with_text(system_prompt, tokenizer))
        } else {
            None
        };

        let conversation = Conversation {
            chat_id: chat_id.0 as u64,
            turns: Default::default(),
            prompt_tokens: 0,
            is_authorized,
            openai_api_key: open_ai_api_key,
            system_prompt,
        };

        let history = {
            // Fetch latest messages first so we can stop once the token budget is exceeded,
            // then restore chronological order for conversation reconstruction.
            let mut stmt = conn.prepare(
                "SELECT tokens, role, text FROM history WHERE chat_id = ?1 ORDER BY id DESC",
            )?;

            let rows = stmt.query_map([chat_id.0], |row| {
                let tokens: usize = row.get(0)?;
                let role: u8 = row.get(1)?;
                let text: String = row.get(2)?;
                Ok((
                    MessageRole::try_from(role).expect("Invalid message role"),
                    conversation::Message::with_text_and_tokens(text, tokens),
                ))
            })?;

            let mut history: Vec<(MessageRole, conversation::Message)> = Vec::new();
            let mut total_tokens: usize = 0;

            for row in rows {
                if let Ok((role, message)) = row {
                    // Stop before adding a message that would push us over the budget.
                    if total_tokens + message.tokens > max_tokens {
                        break;
                    }
                    total_tokens += message.tokens;
                    history.push((role, message));
                }
            }

            history
        };

        (conversation, history)
    };

    let mut user_message: Option<conversation::Message> = None;

    // We iterated newest-to-oldest; restore to oldest-first.
    for (role, message) in history.into_iter().rev() {
        conversation.prompt_tokens += message.tokens;

        match role {
            MessageRole::User => {
                user_message = Some(message);
            }
            MessageRole::Assistant => {
                conversation.add_turn(ChatTurn {
                    user: user_message.take().expect("Missing user message"),
                    assistant: message,
                });
            }
            _ => {
                log::error!("Invalid message role in DB for chat {}", chat_id);
                panic!("Invalid message role");
            }
        }
    }

    // Drop a trailing user message without assistant to avoid panic on malformed history.
    if let Some(unpaired) = user_message.take() {
        log::warn!(
            "Dropping trailing user message without assistant response for chat {}",
            chat_id
        );
        conversation.prompt_tokens = conversation.prompt_tokens.saturating_sub(unpaired.tokens);
    }

    log::info!(
        "Loaded conversation {} with {} messages and {} tokens",
        conversation.chat_id,
        conversation.turns.len() * 2,
        conversation.prompt_tokens
    );

    Ok(conversation)
}

pub async fn add_chat_turn(
    db: &Arc<Mutex<Connection>>,
    chat_id: ChatId,
    turn: ChatTurn,
) -> anyhow::Result<()> {
    // Ensure both user and assistant messages are persisted atomically.
    let mut conn = db.lock().await;
    let tx = conn.transaction()?;

    tx.execute(
        "INSERT INTO history (chat_id, tokens, role, text) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![
            chat_id.0,
            turn.user.tokens as i64,
            MessageRole::User as u8,
            turn.user.text
        ],
    )?;

    tx.execute(
        "INSERT INTO history (chat_id, tokens, role, text) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![
            chat_id.0,
            turn.assistant.tokens as i64,
            MessageRole::Assistant as u8,
            turn.assistant.text
        ],
    )?;

    tx.commit()?;

    log::info!("Added chat turn to conversation {}", chat_id);

    Ok(())
}
