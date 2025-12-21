use crate::conversation::{self, Conversation, Message, MessageRole};
use crate::openrouter_api;
use crate::panic_handler::fatal_panic;
use teloxide::types::ChatId;
use tokio_rusqlite::Connection;
use tokio_rusqlite::rusqlite::{Connection as SyncConnection, Error as SqliteError, params};

const SCHEMA_VERSION: i32 = 1;

pub async fn init_db() -> Connection {
    let db_path = std::env::var("SQLITE_PATH").unwrap_or_else(|_| "data/db.sqlite".to_string());

    // Ensure parent directory exists
    if let Some(parent) = std::path::Path::new(&db_path).parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).expect("failed to create parent directory");
    }

    let conn = Connection::open(&db_path)
        .await
        .expect("failed to open database");

    conn.call(|conn| {
        match std::env::var("DB_ENCRYPTION_KEY") {
            Ok(key) if !key.is_empty() => conn
                .pragma_update(None, "key", &key)
                .expect("failed to set database encryption key pragma"),
            _ => log::warn!("DB_ENCRYPTION_KEY not set; database will be unencrypted"),
        }

        // Initialize database schema if needed and validate version.
        let version = get_schema_version(conn);
        if version == 0 {
            init_schema(conn);
            set_schema_version(conn, SCHEMA_VERSION);
            log::info!("Initialized database schema version {}", SCHEMA_VERSION);
        } else if version == SCHEMA_VERSION {
            log::info!("Database schema version {} detected", version);
        } else {
            fatal_panic(format!(
                "Unsupported database schema version {} (expected {})",
                version, SCHEMA_VERSION
            ));
        }

        Ok::<(), SqliteError>(())
    })
    .await
    .expect("failed to initialize database");

    conn
}

fn init_schema(conn: &SyncConnection) {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     INTEGER NOT NULL,
            role        INTEGER NOT NULL,
            text        TEXT NOT NULL
        )",
        [],
    )
    .expect("failed to create history table");

    conn.execute(
        "CREATE TABLE IF NOT EXISTS chats (
            chat_id                 INTEGER PRIMARY KEY NOT NULL,
            is_authorized           BOOLEAN NOT NULL DEFAULT 0,
            is_admin                BOOLEAN NOT NULL DEFAULT 0,
            openrouter_api_key      TEXT,
            model_id                TEXT,
            system_prompt           TEXT,
            user_name               TEXT
        )",
        [],
    )
    .expect("failed to create chats table");
}

fn get_schema_version(conn: &SyncConnection) -> i32 {
    conn.query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap_or_default()
}

fn set_schema_version(conn: &SyncConnection, version: i32) {
    conn.pragma_update(None, "user_version", version)
        .expect("failed to set schema version");
}

pub async fn load_conversation(db: &Connection, chat_id: ChatId) -> Conversation {
    let chat_id_val = chat_id.0;

    db.call(move |conn| {
            // Fetch exactly one chat row; panic if multiple rows are found.
            let (is_authorized, is_admin, openrouter_api_key, model_id, system_prompt, user_name) = conn
                .query_row(
                    "SELECT is_authorized, is_admin, openrouter_api_key, model_id, system_prompt, user_name FROM chats WHERE chat_id = ?1",
                    [chat_id_val],
                    |row| {
                        Ok((
                            row.get::<_, bool>(0)?,
                            row.get::<_, bool>(1)?,
                            row.get::<_, Option<String>>(2)?,
                            row.get::<_, Option<String>>(3)?,
                            row.get::<_, Option<String>>(4)?,
                            row.get::<_, Option<String>>(5)?,
                        ))
                    },
                )
                .or_else(|err| {
                    if matches!(err, tokio_rusqlite::rusqlite::Error::QueryReturnedNoRows) {
                        let r = conn
                            .execute(
                                "INSERT INTO chats (chat_id, is_authorized, is_admin, openrouter_api_key, model_id, system_prompt, user_name) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                                params![
                                    chat_id_val,
                                    false,
                                    false,
                                    Option::<String>::None,
                                    Option::<String>::None,
                                    Option::<String>::None,
                                    Option::<String>::None
                                ],
                            )
                            .expect("failed to insert chat row");
                        if r != 1 {
                            fatal_panic(format!(
                                "failed to insert chat row for chat_id {}",
                                chat_id.0
                            ));
                        }
                        Ok((false, false, None, None, None, None))
                    } else {
                        Err(err)
                    }
                })
                .expect("failed to fetch chat row");

            let system_prompt = system_prompt
                .filter(|s| !s.is_empty())
                .map(|text| conversation::Message {
                    role: MessageRole::System,
                    text,
                });

            Ok::<Conversation, SqliteError>(Conversation {
                chat_id: chat_id_val,
                history: Default::default(),
                is_authorized,
                is_admin,
                openrouter_api_key,
                model_id,
                system_prompt,
                user_name,
            })
        })
        .await
        .expect("failed to load conversation")
}

pub async fn load_history(db: &Connection, conversation: &mut Conversation, token_budget: u64) {
    conversation.history.clear();

    let chat_id = conversation.chat_id;

    let messages: Vec<(u8, String)> = db
        .call(move |conn| {
            let mut stmt = conn
                .prepare("SELECT role, text FROM history WHERE chat_id = ?1 ORDER BY id DESC")
                .expect("failed to prepare history lookup statement");

            let rows = stmt
                .query_map([chat_id], |row| {
                    let role: u8 = row.get(0)?;
                    let text: String = row.get(1)?;
                    Ok((role, text))
                })
                .expect("failed to query history rows");

            let mut collected = Vec::new();
            for row in rows {
                collected.push(row.expect("failed to read history row"));
            }
            Ok::<Vec<(u8, String)>, SqliteError>(collected)
        })
        .await
        .expect("failed to load history rows");

    for (role_raw, text) in messages {
        let role = MessageRole::try_from(role_raw).expect("invalid message role");
        conversation
            .history
            .push_front(conversation::Message { role, text });
        let estimated_tokens =
            openrouter_api::estimate_tokens(conversation.history.iter().map(|m| m.text.as_str()));
        if estimated_tokens > token_budget {
            break;
        }
    }
}

pub async fn add_messages<I>(db: &Connection, chat_id: ChatId, messages: I)
where
    I: IntoIterator<Item = Message>,
{
    let messages: Vec<Message> = messages.into_iter().collect();

    db.call(move |conn| {
        let tx = conn.transaction().expect("failed to start transaction");

        for msg in messages {
            tx.execute(
                "INSERT INTO history (chat_id, role, text) VALUES (?1, ?2, ?3)",
                params![chat_id.0, msg.role as u8, msg.text],
            )
            .expect("failed to insert message");
        }

        tx.commit().expect("failed to commit transaction");

        log::info!("Added chat turn to conversation {}", chat_id);
        Ok::<(), SqliteError>(())
    })
    .await
    .expect("failed to add messages");
}

pub async fn set_openrouter_api_key(
    db: &Connection,
    chat_id: ChatId,
    openrouter_api_key: Option<&str>,
) {
    let openrouter_api_key = openrouter_api_key.map(|s| s.to_owned());

    let updated = db
        .call(move |conn| {
            conn.execute(
                "UPDATE chats SET openrouter_api_key = ?2 WHERE chat_id = ?1",
                params![chat_id.0, openrouter_api_key],
            )
        })
        .await
        .expect("failed to update api key");

    if updated != 1 {
        fatal_panic(format!(
            "failed to update api key for chat_id {} (updated {})",
            chat_id.0, updated
        ));
    }
}

pub async fn set_model_id(db: &Connection, chat_id: ChatId, model_id: Option<&str>) {
    let model_id = model_id.map(|s| s.to_owned());

    let updated = db
        .call(move |conn| {
            conn.execute(
                "UPDATE chats SET model_id = ?2 WHERE chat_id = ?1",
                params![chat_id.0, model_id],
            )
        })
        .await
        .expect("failed to update model id");

    if updated != 1 {
        fatal_panic(format!(
            "failed to update model id for chat_id {} (updated {})",
            chat_id.0, updated
        ));
    }
}

pub async fn set_system_prompt(db: &Connection, chat_id: ChatId, system_prompt: Option<&str>) {
    let system_prompt = system_prompt.map(|s| s.to_owned());

    let updated = db
        .call(move |conn| {
            conn.execute(
                "UPDATE chats SET system_prompt = ?2 WHERE chat_id = ?1",
                params![chat_id.0, system_prompt],
            )
        })
        .await
        .expect("failed to update system prompt");

    if updated != 1 {
        fatal_panic(format!(
            "failed to update system prompt for chat_id {} (updated {})",
            chat_id.0, updated
        ));
    }
}

pub async fn set_user_name(db: &Connection, chat_id: ChatId, user_name: Option<&str>) {
    let user_name = user_name.map(|s| s.to_owned());

    let updated = db
        .call(move |conn| {
            conn.execute(
                "UPDATE chats SET user_name = ?2 WHERE chat_id = ?1",
                params![chat_id.0, user_name],
            )
        })
        .await
        .expect("failed to update user name");

    if updated != 1 {
        fatal_panic(format!(
            "failed to update user name for chat_id {} (updated {})",
            chat_id.0, updated
        ));
    }
}

pub async fn set_is_authorized(
    db: &Connection,
    chat_id: ChatId,
    is_authorized: bool,
) -> anyhow::Result<()> {
    let updated = db
        .call(move |conn| {
            conn.execute(
                "UPDATE chats SET is_authorized = ?2 WHERE chat_id = ?1",
                params![chat_id.0, is_authorized],
            )
        })
        .await
        .expect("failed to update is_authorized");

    if updated == 1 {
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "failed to update is_authorized for chat_id {}",
            chat_id.0
        ))
    }
}

pub async fn list_unauthorized_chats(db: &Connection) -> Vec<(i64, Option<String>)> {
    db.call(|conn| {
        let mut stmt = conn
            .prepare(
                "SELECT chat_id, user_name FROM chats WHERE is_authorized = 0 ORDER BY chat_id",
            )
            .expect("failed to prepare unauthorized chats query");

        let rows = stmt
            .query_map([], |row| {
                let chat_id: i64 = row.get(0)?;
                let user_name: Option<String> = row.get(1)?;
                Ok((chat_id, user_name))
            })
            .expect("failed to query unauthorized chats");

        let mut collected = Vec::new();
        for row in rows {
            collected.push(row.expect("failed to read unauthorized chat row"));
        }
        Ok::<Vec<(i64, Option<String>)>, SqliteError>(collected)
    })
    .await
    .expect("failed to list unauthorized chats")
}
