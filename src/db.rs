use crate::conversation::{self, Conversation, Message, MessageRole, TokenCounter};
use anyhow::{Context, Result, anyhow};
use rusqlite::{Connection, Error as SqliteError};
use std::sync::Arc;
use teloxide::types::ChatId;
use tokio::sync::Mutex;

const SCHEMA_VERSION: i32 = 2;

pub fn init_db() -> Connection {
    let db_path = std::env::var("SQLITE_PATH").unwrap_or_else(|_| "data/db.sqlite".to_string());

    // Ensure parent directory exists
    if let Some(parent) = std::path::Path::new(&db_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("Failed to create parent directory");
        }
    }

    let conn = Connection::open(&db_path).expect("Failed to open database");

    match std::env::var("DB_ENCRYPTION_KEY") {
        Ok(key) if !key.is_empty() => conn
            .pragma_update(None, "key", &key)
            .expect("Failed to set encryption key"),
        _ => log::warn!("DB_ENCRYPTION_KEY not set; database will be unencrypted"),
    }

    // Initialize database schema if needed and validate version.
    let version = get_schema_version(&conn);
    match version {
        0 => {
            init_schema(&conn);
            set_schema_version(&conn, SCHEMA_VERSION);
            log::info!("Initialized database schema version {}", SCHEMA_VERSION);
        }
        1 => {
            // Migrate from version 1 to version 2
            conn.execute("ALTER TABLE history ADD COLUMN raw_text TEXT NULL", [])
                .expect("Failed to add raw_text column");
            conn.execute(
                "ALTER TABLE history ADD COLUMN send_failed BOOLEAN NOT NULL DEFAULT 0",
                [],
            )
            .expect("Failed to add send_failed column");
            set_schema_version(&conn, SCHEMA_VERSION);
            log::info!("Migrated database schema from version 1 to version 2");
        }
        2 => log::info!("Database schema version {} detected", version),
        _ => panic!(
            "Unsupported database schema version {} (expected {})",
            version, SCHEMA_VERSION
        ),
    }

    conn
}

fn init_schema(conn: &Connection) {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL,
            tokens  INTEGER NOT NULL,
            role    INTEGER NOT NULL,
            text    TEXT NOT NULL,
            raw_text TEXT NULL,
            send_failed BOOLEAN NOT NULL DEFAULT 0
        )",
        [],
    )
    .expect("Failed to create history table");

    conn.execute(
        "CREATE TABLE IF NOT EXISTS chats (
            chat_id INTEGER PRIMARY KEY NOT NULL,
            is_authorized BOOLEAN NOT NULL,
            open_ai_api_key  TEXT NOT NULL,
            system_prompt    TEXT NOT NULL
        )",
        [],
    )
    .expect("Failed to create chats table");
}

fn get_schema_version(conn: &Connection) -> i32 {
    conn.query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("Failed to get schema version")
}

fn set_schema_version(conn: &Connection, version: i32) {
    conn.pragma_update(None, "user_version", version)
        .expect("Failed to set schema version");
}

pub async fn load_conversation(
    db: &Arc<Mutex<Connection>>,
    chat_id: ChatId,
    tokenizer: &TokenCounter,
    max_tokens: usize,
) -> Conversation {
    let (mut conversation, history) = {
        let conn = db.lock().await;

        let conversation = {
            // Fetch exactly one chat row; panic if multiple rows are found.
            let mut stmt = conn
                .prepare(
                    "SELECT is_authorized, open_ai_api_key, system_prompt FROM chats WHERE chat_id = ?1 LIMIT 2",
                )
                .expect("Failed to prepare statement");
            let mut rows = stmt.query([chat_id.0]).expect("Failed to query chat row");

            let (is_authorized, open_ai_api_key, system_prompt) = match rows
                .next()
                .expect("Failed to fetch chat row")
            {
                Some(row) => {
                    let is_authorized: bool = row.get(0).expect("Failed to fetch is_authorized");
                    let open_ai_api_key: String =
                        row.get(1).expect("Failed to fetch open_ai_api_key");
                    let system_prompt: String = row.get(2).expect("Failed to fetch system_prompt");
                    (is_authorized, open_ai_api_key, system_prompt)
                }
                None => {
                    let r = conn.execute(
                    "INSERT INTO chats (chat_id, is_authorized, open_ai_api_key, system_prompt) VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![chat_id.0, false, "", ""],
                ).expect("Failed to insert chat row");

                    if r != 1 {
                        let error = format!("failed to insert chat row for chat_id {}", chat_id.0);
                        log::error!("{}", error);
                        panic!("{}", error);
                    }

                    (false, String::new(), String::new())
                }
            };

            if rows.next().expect("Failed to fetch next row").is_some() {
                panic!("multiple chat rows found for chat_id {}", chat_id.0);
            }

            let system_prompt = if !system_prompt.is_empty() {
                Some(conversation::Message {
                    role: MessageRole::System,
                    tokens: tokenizer.count_text(&system_prompt),
                    text: system_prompt,
                    ..Default::default()
                })
            } else {
                None
            };

            Conversation {
                chat_id: chat_id.0,
                history: Default::default(),
                prompt_tokens: 0,
                is_authorized,
                openai_api_key: open_ai_api_key,
                system_prompt,
                command: None,
            }
        };

        let history =
            {
                // Fetch latest messages first so we can stop once the token budget is exceeded,
                // then restore chronological order for conversation reconstruction.
                let mut stmt = conn.prepare(
                "SELECT tokens, role, text FROM history WHERE chat_id = ?1 ORDER BY id DESC",
            ).expect("Failed to prepare statement");

                let rows = stmt
                    .query_map([chat_id.0], |row| {
                        let tokens: usize = row.get(0).expect("Failed to get tokens");
                        let role: u8 = row.get(1).expect("Failed to get role");
                        let text: String = row.get(2).expect("Failed to get text");
                        let role = MessageRole::try_from(role).expect("Invalid message role");

                        Ok(conversation::Message {
                            role,
                            tokens,
                            text,
                            ..Default::default()
                        })
                    })
                    .expect("Failed to query history");

                let mut history: Vec<conversation::Message> = Vec::new();
                let mut total_tokens: usize = 0;

                for row in rows {
                    if let Ok(message) = row {
                        // Stop before adding a message that would push us over the budget.
                        if total_tokens + message.tokens > max_tokens {
                            break;
                        }
                        total_tokens += message.tokens;
                        history.push(message);
                    }
                }

                history
            };

        (conversation, history)
    };

    let mut user_message: Option<conversation::Message> = None;

    // We collected newest-to-oldest; process oldest-first while respecting the token budget.
    for message in history.into_iter().rev() {
        conversation.prompt_tokens += message.tokens;

        match message.role {
            MessageRole::User => {
                user_message = Some(message);
            }
            MessageRole::Assistant => {
                conversation.add_message(user_message.take().expect("Missing user message"));
                conversation.add_message(message);
            }
            _ => {
                log::error!("Invalid message role in DB for chat {}", chat_id);
                panic!("Invalid message role");
            }
        }
    }

    if user_message.is_some() {
        let error = format!(
            "Dropping trailing user message without assistant response for chat {}",
            chat_id
        );
        log::error!("{}", error);
        panic!("{}", error)
    }

    log::info!(
        "Loaded conversation {} with {} messages and {} tokens",
        conversation.chat_id,
        conversation.history.len() * 2,
        conversation.prompt_tokens
    );

    conversation
}

pub async fn add_messages<'a, I>(db: &Arc<Mutex<Connection>>, chat_id: ChatId, messages: I)
where
    I: IntoIterator<Item = Message>,
{
    // Ensure all messages are persisted atomically.
    let mut conn = db.lock().await;
    let tx = conn.transaction().expect("Failed to start transaction");

    let mut msg_count = 0;
    for msg in messages {
        tx.execute(
            "INSERT INTO history (chat_id, tokens, role, text, raw_text, send_failed) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                chat_id.0,
                msg.tokens as i64,
                msg.role as u8,
                msg.text,
                msg.raw_text,
                msg.send_failed
            ],
        )
        .expect("Failed to insert message");
        msg_count += 1;
    }

    tx.commit().expect("Failed to commit transaction");

    log::info!("Added {} messages to conversation {}", msg_count, chat_id);
}

pub async fn update_token(db: &Arc<Mutex<Connection>>, chat_id: i64, token: String) -> Result<()> {
    let conn = db.lock().await;
    let updated = conn
        .execute(
            "UPDATE chats SET open_ai_api_key = ?1 WHERE chat_id = ?2",
            rusqlite::params![token, chat_id],
        )
        .context("Failed to update token")?;

    if updated != 1 {
        return Err(anyhow!(
            "Token update affected {} rows for chat_id {}",
            updated,
            chat_id
        ));
    }

    log::info!("Updated OpenAI token for chat {}", chat_id);
    Ok(())
}

pub async fn update_system_message(
    db: &Arc<Mutex<Connection>>,
    chat_id: i64,
    system_message: String,
) -> Result<()> {
    let conn = db.lock().await;
    let updated = conn
        .execute(
            "UPDATE chats SET system_prompt = ?1 WHERE chat_id = ?2",
            rusqlite::params![system_message, chat_id],
        )
        .context("Failed to update system message")?;

    if updated != 1 {
        return Err(anyhow!(
            "System message update affected {} rows for chat_id {}",
            updated,
            chat_id
        ));
    }

    log::info!("Updated system prompt for chat {}", chat_id);
    Ok(())
}
