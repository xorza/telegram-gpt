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

    let mut conv = Conversation {
        chat_id: chat_id.0 as u64,
        turns: Default::default(),
        prompt_tokens: 0,
        is_authorized,
        openai_api_key: open_ai_api_key,
        system_prompt: Some(conversation::Message::with_text(system_prompt, tokenizer)),
    };

    {
        let mut stmt = conn
            .prepare("SELECT tokens, role, text FROM history WHERE chat_id = ?1 ORDER BY id ASC")?;

        let rows = stmt.query_map([chat_id.0], |row| {
            let tokens: usize = row.get(0)?;
            let role: u8 = row.get(1)?;
            let text: String = row.get(2)?;
            Ok((
                MessageRole::try_from(role).expect("Invalid message role"),
                conversation::Message::with_text_and_tokens(text, tokens),
            ))
        })?;

        let mut user_message: Option<conversation::Message> = None;
        for row in rows {
            if let Ok((role, message)) = row {
                conv.prompt_tokens += message.tokens;

                match role {
                    MessageRole::User => {
                        user_message = Some(message);
                    }
                    MessageRole::Assistant => {
                        conv.add_turn(ChatTurn {
                            user: user_message.take().expect("No user message found"),
                            assistant: message,
                        });
                    }
                    _ => {
                        let error = "Invalid message role";
                        log::error!("{}", error);
                        panic!("{}", error)
                    }
                }
            }
        }

        if user_message.is_some() {
            let error = "Last user message not followed by assistant response";
            log::error!("{}", error);
            panic!("{}", error);
        }
    }

    log::info!(
        "Loaded conversation {} with {} messages and {} tokens",
        conv.chat_id,
        conv.turns.len() * 2,
        conv.prompt_tokens
    );

    Ok(conv)
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
