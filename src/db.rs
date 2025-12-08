use anyhow::Result;
use rusqlite::{Connection, Error as SqliteError};

pub fn init_db() -> Result<Connection> {
    let db_path = std::env::var("SQLITE_PATH").unwrap_or_else(|_| "data/bot.db".to_string());

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

    // Initialize database schema
    init_schema(&conn)?;

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
