use crate::conversation::{Conversation, Message, MessageRole, TokenCounter};
use anyhow::Result;
use rusqlite::{Connection, Error as SqliteError};
use teloxide::types::ChatId;

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
    let _ = conn.pragma_update(None, "journal_mode", "WAL");

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

pub fn load_conversation(chat_id: ChatId, conn: &Connection) -> anyhow::Result<Conversation> {
    let (is_authorized, _open_ai_api_key, _system_prompt) = {
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
        next_turn_id: 0,
        chat_id: chat_id.0 as u64,
        turns: Default::default(),
        prompt_tokens: 0,
        is_authorized,
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
                Message::with_text_and_tokens(text, tokens),
            ))
        })?;

        let mut last_user_turn: Option<u64> = None;
        for row in rows {
            if let Ok((role, message)) = row {
                conv.prompt_tokens += message.tokens;

                match role {
                    MessageRole::User => {
                        let turn_id = conv.record_user_message(message);
                        last_user_turn = Some(turn_id);
                    }
                    MessageRole::Assistant => {
                        conv.record_assistant_response(
                            last_user_turn.expect("No user message found"),
                            message,
                        );
                        last_user_turn = None;
                    }
                    _ => {
                        let error = "Invalid message role";
                        log::error!("{}", error);
                        panic!("{}", error)
                    }
                }
            }
        }

        if last_user_turn.is_some() {
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
