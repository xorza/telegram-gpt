use crate::conversation::{self, Conversation, Message, MessageRole};
use crate::openrouter_api;
use anyhow::Result;
use rusqlite::{Connection, Error as SqliteError};
use std::sync::Arc;
use teloxide::types::ChatId;
use tokio::sync::Mutex;

const SCHEMA_VERSION: i32 = 1;

pub fn init_db() -> Connection {
    let db_path = std::env::var("SQLITE_PATH").unwrap_or_else(|_| "data/db.sqlite".to_string());

    // Ensure parent directory exists
    if let Some(parent) = std::path::Path::new(&db_path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).expect("failed to create parent directory");
    }

    let conn = Connection::open(&db_path).expect("failed to open database");

    match std::env::var("DB_ENCRYPTION_KEY") {
        Ok(key) if !key.is_empty() => conn
            .pragma_update(None, "key", &key)
            .expect("failed to set database encryption key pragma"),
        _ => log::warn!("DB_ENCRYPTION_KEY not set; database will be unencrypted"),
    }

    // Initialize database schema if needed and validate version.
    let version = get_schema_version(&conn);
    if version == 0 {
        init_schema(&conn);
        set_schema_version(&conn, SCHEMA_VERSION);
        log::info!("Initialized database schema version {}", SCHEMA_VERSION);
    } else if version != SCHEMA_VERSION {
        panic!(
            "Unsupported database schema version {} (expected {})",
            version, SCHEMA_VERSION
        );
    } else {
        log::info!("Database schema version {} detected", version);
    }

    conn
}

fn init_schema(conn: &Connection) {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            role    INTEGER NOT NULL,
            text    TEXT NOT NULL
        )",
        [],
    )
    .expect("failed to create history table");

    conn.execute(
        "CREATE TABLE IF NOT EXISTS chats (
            chat_id INTEGER PRIMARY KEY NOT NULL,
            is_authorized BOOLEAN NOT NULL,
            openrouter_api_key  TEXT NOT NULL,
            system_prompt    TEXT NOT NULL
        )",
        [],
    )
    .expect("failed to create chats table");
}

fn get_schema_version(conn: &Connection) -> i32 {
    conn.query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("failed to get schema version")
}

fn set_schema_version(conn: &Connection, version: i32) {
    conn.pragma_update(None, "user_version", version)
        .expect("failed to set schema version");
}

pub async fn load_conversation(
    db: &Arc<Mutex<Connection>>,
    chat_id: ChatId,
    token_budget: u64,
) -> Conversation {
    let (mut conversation, history) = {
        let conn = db.lock().await;

        let conversation = {
            // Fetch exactly one chat row; panic if multiple rows are found.
            let mut stmt = conn
                .prepare(
                    "SELECT is_authorized, openrouter_api_key, system_prompt \
            FROM chats WHERE chat_id = ?1 LIMIT 2",
                )
                .expect("failed to prepare chat lookup statement");
            let mut rows = stmt.query([chat_id.0]).expect("failed to query chat row");

            let (is_authorized, openrouter_api_key, system_prompt) = match rows
                .next()
                .expect("failed to read chat row")
            {
                Some(row) => {
                    let is_authorized: bool = row.get(0).expect("failed to get is_authorized");
                    let openrouter_api_key: String =
                        row.get(1).expect("failed to get openrouter_api_key");
                    let system_prompt: String = row.get(2).expect("failed to get system_prompt");
                    (is_authorized, openrouter_api_key, system_prompt)
                }
                None => {
                    let r = conn.execute(
                        "INSERT INTO chats (chat_id, is_authorized, openrouter_api_key, system_prompt) \
                        VALUES (?1, ?2, ?3, ?4)",
                        rusqlite::params![chat_id.0, false, "", ""],
                    ).expect("failed to insert chat row");
                    if r != 1 {
                        let error = format!("failed to insert chat row for chat_id {}", chat_id.0);
                        log::error!("{}", error);
                        panic!("{}", error);
                    }

                    (false, String::new(), String::new())
                }
            };

            let system_prompt = if !system_prompt.is_empty() {
                Some(conversation::Message {
                    role: MessageRole::System,
                    text: system_prompt,
                })
            } else {
                None
            };

            Conversation {
                chat_id: chat_id.0 as u64,
                history: Default::default(),

                is_authorized,
                openai_api_key: openrouter_api_key,
                system_prompt,
            }
        };

        let history = {
            // Fetch latest messages first so we can stop once the token budget is exceeded,
            // then restore chronological order for conversation reconstruction.
            let mut stmt = conn
                .prepare("SELECT role, text FROM history WHERE chat_id = ?1 ORDER BY id DESC")
                .expect("failed to prepare history lookup statement");

            let mut history: Vec<conversation::Message> = Vec::new();

            let rows = stmt
                .query_map([chat_id.0], |row| {
                    let role: u8 = row.get(0)?;
                    let text: String = row.get(1)?;
                    let role = MessageRole::try_from(role).expect("Invalid message role");

                    Ok(conversation::Message { role, text })
                })
                .expect("failed to query history rows")
                .filter_map(|row| row.ok());
            for message in rows {
                history.push(message);

                let estimated_tokens =
                    openrouter_api::estimate_tokens(history.iter().map(|m| m.text.as_str()));
                if estimated_tokens > token_budget {
                    break;
                }
            }

            history
        };

        (conversation, history)
    };

    // We collected newest-to-oldest; process oldest-first while respecting the token budget.
    for message in history.into_iter().rev() {
        match message.role {
            MessageRole::User => {
                conversation.add_message(message);
            }
            MessageRole::Assistant => {
                conversation.add_message(message);
            }
            _ => {
                log::error!("Invalid message role in DB for chat {}", chat_id);
                panic!("Invalid message role");
            }
        }
    }

    log::info!(
        "Loaded conversation {} with {} messages",
        conversation.chat_id,
        conversation.history.len(),
    );

    conversation
}

pub async fn add_messages<I>(db: &Arc<Mutex<Connection>>, chat_id: ChatId, messages: I)
where
    I: IntoIterator<Item = Message>,
{
    // Ensure both user and assistant messages are persisted atomically.
    let mut conn = db.lock().await;
    let tx = conn.transaction().expect("failed to start transaction");

    for msg in messages {
        tx.execute(
            "INSERT INTO history (chat_id, role, text) VALUES (?1, ?2, ?3)",
            rusqlite::params![chat_id.0, msg.role as u8, msg.text],
        )
        .expect("failed to insert message");
    }

    tx.commit().expect("failed to commit transaction");

    log::info!("Added chat turn to conversation {}", chat_id);
}
