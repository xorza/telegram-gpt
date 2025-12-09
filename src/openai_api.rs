use std::panic::AssertUnwindSafe;

use crate::DynError;
use crate::conversation::{Conversation, Message, MessageRole};
use anyhow::{Context, anyhow};
use futures_util::StreamExt;
use reqwest::Client;
use rusqlite::OpenFlags;
use serde_json::json;

#[derive(Debug)]
enum ContentType {
    Input,
    Output,
}

pub fn prepare_payload<'a, I>(model: &str, messages: I, stream: bool) -> serde_json::Value
where
    I: IntoIterator<Item = &'a Message>,
{
    let mut input_items = Vec::new();

    for msg in messages {
        input_items.push(text_content(
            msg.role,
            &msg.text,
            if msg.role == MessageRole::Assistant {
                ContentType::Output
            } else {
                ContentType::Input
            },
        ));
    }

    json!({
        "model": model,
        "input": input_items,
        "tools": [
            {
                "type": "web_search",
                // "user_location": {
                //     "type": "approximate",
                //     "country": "US"
                // }
            }
        ],
        "tool_choice": "auto",
        "stream": stream,
    })
}

#[allow(dead_code)]
pub async fn send<F, Fut>(
    http: &Client,
    api_key: &str,
    payload: serde_json::Value,
    _stream: bool,
    mut on_delta: F,
) -> anyhow::Result<String, DynError>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>> + Send,
{
    let response = http
        .post("https://api.openai.com/v1/responses")
        .bearer_auth(api_key)
        .json(&payload)
        .send()
        .await?;

    let status = response.status();
    let body_text = response.text().await?;

    if !status.is_success() {
        return Err(format!("OpenAI Responses API error {status}: {body_text}").into());
    }

    let response_body: serde_json::Value = serde_json::from_str(&body_text)?;

    if let Some(text) = extract_output_text(&response_body) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            let owned = trimmed.to_string();
            on_delta(owned.clone()).await?;
            return Ok(owned);
        }
    }

    Err(format!("OpenAI response missing text output: {response_body}").into())
}

#[allow(dead_code)]
pub async fn send_stream<F, Fut>(
    http: &Client,
    api_key: &str,
    payload: serde_json::Value,
    mut on_delta: F,
) -> anyhow::Result<(), DynError>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>> + Send,
{
    let response = http
        .post("https://api.openai.com/v1/responses")
        .bearer_auth(api_key)
        .json(&payload)
        .send()
        .await?;

    let status = response.status();
    if !status.is_success() {
        let body_text = response.text().await?;
        return Err(format!("OpenAI Responses API error {status}: {body_text}").into());
    }

    let mut stream = response.bytes_stream();
    let mut buffer: Vec<u8> = Vec::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.extend_from_slice(&chunk);

        while let Some(event) = pop_next_event(&mut buffer) {
            let event_text = String::from_utf8_lossy(&event);

            for line in event_text.lines() {
                let line = line.trim();
                if !line.starts_with("data:") {
                    continue;
                }

                let data = line.trim_start_matches("data:").trim();

                // Streaming sentinel from OpenAI
                if data == "[DONE]" {
                    return Ok(());
                }

                let value: serde_json::Value = serde_json::from_str(data)?;

                if let Some(delta) = extract_stream_delta(&value) {
                    if !delta.is_empty() {
                        on_delta(delta).await?;
                    }
                }
            }
        }
    }

    Ok(())
}

fn text_content(role: MessageRole, text: &str, content_type: ContentType) -> serde_json::Value {
    let type_str = match content_type {
        ContentType::Input => "input_text",
        ContentType::Output => "output_text",
    };

    json!({
        "role": role.to_string(),
        "content": [
            {
                "type": type_str,
                "text": text
            }
        ]
    })
}

fn extract_output_text(value: &serde_json::Value) -> Option<String> {
    if let Some(array) = value.get("output_text").and_then(|v| v.as_array()) {
        if !array.is_empty() {
            return Some(
                array
                    .iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
            );
        }
    }

    if let Some(output) = value.get("output").and_then(|v| v.as_array()) {
        let mut chunks = Vec::new();
        for item in output {
            if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                for part in content {
                    if part.get("type").and_then(|t| t.as_str()) == Some("output_text") {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            chunks.push(text);
                        }
                    } else if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            chunks.push(text);
                        }
                    }
                }
            }
        }
        if !chunks.is_empty() {
            return Some(chunks.join("\n"));
        }
    }

    None
}

fn pop_next_event(buffer: &mut Vec<u8>) -> Option<Vec<u8>> {
    fn find_separator(buf: &[u8]) -> Option<(usize, usize)> {
        let mut i = 0;
        while i + 1 < buf.len() {
            if i + 3 < buf.len() && &buf[i..i + 4] == b"\r\n\r\n" {
                return Some((i, 4));
            }
            if &buf[i..i + 2] == b"\n\n" {
                return Some((i, 2));
            }
            i += 1;
        }
        None
    }

    let (idx, sep_len) = find_separator(buffer)?;
    let event: Vec<u8> = buffer.drain(..idx).collect();
    buffer.drain(..sep_len);
    Some(event)
}

fn extract_stream_delta(value: &serde_json::Value) -> Option<String> {
    if let Some(delta) = value.get("delta").and_then(|d| d.as_str()) {
        if !delta.is_empty() {
            return Some(delta.to_string());
        }
    }

    if let Some(output) = value.get("output").and_then(|v| v.as_array()) {
        let mut chunks = Vec::new();
        for item in output {
            if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
                for part in content {
                    if part.get("type").and_then(|t| t.as_str()) == Some("text_delta") {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            chunks.push(text);
                        }
                    } else if let Some(delta) = part.get("delta").and_then(|d| d.as_str()) {
                        chunks.push(delta);
                    }
                }
            }
        }
        if !chunks.is_empty() {
            return Some(chunks.join(""));
        }
    }

    None
}

pub fn context_length(model: &str) -> usize {
    // Values sourced from OpenAI model docs (Dec 8, 2025).
    match model {
        // GPT-5.1 flagship & codex variants (400k)
        m if m.starts_with("gpt-5.1-codex-max")
            || m.starts_with("gpt-5.1-codex")
            || m.starts_with("gpt-5.1-mini")
            || m.starts_with("gpt-5.1") =>
        {
            400_000
        }
        // GPT-5.1 chat has a smaller 128k window
        m if m.starts_with("gpt-5.1-chat") => 128_000,
        // GPT-5 previous generation (400k) and chat (128k)
        m if m.starts_with("gpt-5-chat") => 128_000,
        m if m.starts_with("gpt-5-mini") || m.starts_with("gpt-5") => 400_000,
        // GPT-4.1 family (≈1M)
        m if m.starts_with("gpt-4.1-nano")
            || m.starts_with("gpt-4.1-mini")
            || m.starts_with("gpt-4.1") =>
        {
            1_047_576
        }
        // GPT-4o family (128k) including ChatGPT alias
        m if m.starts_with("gpt-4o") || m.starts_with("chatgpt-4o") => 128_000,
        // Legacy GPT-4
        m if m.starts_with("gpt-4") => 8_192,
        _ => 64_000,
    }
}
