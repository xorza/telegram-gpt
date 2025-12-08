use std::panic::AssertUnwindSafe;

use crate::DynError;
use crate::conversation::{Conversation, MessageRole, TokenizedMessage};
use anyhow::{Context, anyhow};
use reqwest::Client;
use serde_json::{Value, json};

#[derive(Debug)]
enum ContentType {
    Input,
    Output,
}

pub async fn send_with_web_search(
    http: &Client,
    model: &str,
    system_prompt: Option<&TokenizedMessage>,
    conversation: &Conversation,
) -> Result<String, DynError> {
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
        if let Some(assistant) = &turn.assistant {
            input_items.push(text_content(
                MessageRole::Assistant,
                &assistant.text,
                ContentType::Output,
            ));
        }
    }

    if input_items.is_empty() {
        return Err("no content available for OpenAI call".into());
    }

    let payload = json!({
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
    });

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY is required");
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

    let response_body: Value = serde_json::from_str(&body_text)?;

    if let Some(text) = extract_output_text(&response_body) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }

    Err(format!("OpenAI response missing text output: {response_body}").into())
}

fn text_content(role: MessageRole, text: &str, content_type: ContentType) -> Value {
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

fn extract_output_text(value: &Value) -> Option<String> {
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

pub fn context_length(model: &str) -> Option<usize> {
    // Based on OpenAI documentation as of Dec 8, 2025.
    if model.starts_with("gpt-4.1") || model.starts_with("gpt-5.1") {
        return Some(1_000_000); // 1M-token context window
    }
    if model.starts_with("gpt-4o") {
        return Some(128_000); // long-context GPT-4o
    }
    None
}
