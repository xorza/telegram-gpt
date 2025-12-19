# tggpt

Telegram bot that relays user messages to the OpenAI Responses API, keeps a rolling chat history per chat in SQLite, and streams a typing indicator while the model works.

## Features
- Telegram transport via `teloxide`, responding only to text messages.
- Per-chat OpenAI API key, optional system prompt, and on-disk history so context survives restarts.
- Token counting with `tiktoken-rs`; oldest turns are pruned to stay within the model context window.
- Rotating file logs in `logs/` (10 MB, keep 3) plus stdout duplication.

## Prerequisites
- Rust 1.82+ (edition 2024).
- SQLite; `SQLITE_PATH` defaults to `data/db.sqlite` and the directory is created automatically.
- OpenAI API access for each authorized chat.

## Configuration
Set environment variables (e.g., in a `.env` file):

- `TELOXIDE_TOKEN` – Telegram bot token (required).
- `OPEN_AI_MODEL` – OpenAI model name (default: `gpt-4.1`).
- `SQLITE_PATH` – Path to the SQLite database (default: `data/db.sqlite`).
- `DB_ENCRYPTION_KEY` – Optional SQLCipher key if your SQLite build supports it.
- `RUST_LOG` – Optional log level filter (e.g., `info`, `debug`).

## Run
```sh
cargo run --release
```
On first start, the database and `logs/` directory are created automatically.

## Authorizing chats
New chats are inserted into `chats` with `is_authorized = 0` and no API key. The bot will log a warning and ignore messages until the chat is authorized.

1. Send any message to the bot from the chat you want to enable.
2. Find the `chat_id` in the log entry (stored in `logs/` or stdout).
3. Update the database (replace placeholders):
```sh
sqlite3 data/db.sqlite \
  "UPDATE chats SET is_authorized=1, open_ai_api_key='sk-...', system_prompt='You are a helpful assistant.' WHERE chat_id=<chat_id>;"
```

Each chat uses its own OpenAI API key; you can store different keys or prompts per chat.

## Persistence model
- `history` table stores alternating user/assistant messages with token counts.
- `chats` table stores authorization flag, API key, and optional system prompt.
- Conversations are reloaded on startup and trimmed to fit the model's context length.

## Operational notes
- Only text messages are handled; non-text inputs receive a friendly prompt to send text.
- The typing indicator runs while awaiting the OpenAI response and stops once a reply is sent or an error occurs.
- Log rotation may leave up to three compressed history files under `logs/`.
