use std::panic::AssertUnwindSafe;

use crate::DynError;
use crate::conversation::{Conversation, Message, MessageRole};
use anyhow::{Context, anyhow};
use reqwest::Client;
use rusqlite::OpenFlags;
use serde_json::json;

#[derive(Debug)]
enum ContentType {
    Input,
    Output,
}

pub fn prepare_payload(
    model: &str,
    system_prompt: Option<&Message>,
    conversation: &Conversation,
    user_message: &Message,
) -> serde_json::Value {
    let mut input_items = Vec::new();

    if let Some(prompt) = system_prompt {
        input_items.push(text_content(
            MessageRole::System,
            &prompt.text,
            ContentType::Input,
        ));
    }

    for turn in &conversation.turns {
        input_items.push(text_content(
            MessageRole::User,
            &turn.user.text,
            ContentType::Input,
        ));
        input_items.push(text_content(
            MessageRole::Assistant,
            &turn.assistant.text,
            ContentType::Output,
        ));
    }
    input_items.push(text_content(
        MessageRole::User,
        &user_message.text,
        ContentType::Input,
    ));

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
        "tool_choice": "auto"
    })
}

pub async fn send(
    http: &Client,
    api_key: &str,
    payload: serde_json::Value,
) -> anyhow::Result<String, DynError> {
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
            return Ok(trimmed.to_string());
        }
    }

    Err(format!("OpenAI response missing text output: {response_body}").into())
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
