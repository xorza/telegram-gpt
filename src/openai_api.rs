use crate::DynError;
use crate::conversation::{HistoryMessage, MessageRole};
use reqwest::Client;
use serde_json::{Value, json};

pub async fn send_with_web_search(
    http: &Client,
    model: &str,
    system_prompt: Option<&str>,
    messages: &[HistoryMessage],
) -> Result<String, DynError> {
    let mut input_items = Vec::new();

    if let Some(prompt) = system_prompt {
        if !prompt.trim().is_empty() {
            input_items.push(text_content("developer", prompt, ContentType::Input));
        }
    }

    for message in messages {
        let (role, content_type) = match message.role {
            MessageRole::User => ("user", ContentType::Input),
            MessageRole::Assistant => ("assistant", ContentType::Output),
        };

        if !message.text.trim().is_empty() {
            input_items.push(text_content(role, &message.text, content_type));
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
                "user_location": {
                    "type": "approximate",
                    "country": "US"
                }
            }
        ],
        "tool_choice": "auto"
    });

    let api_key = std::env::var("OPENAI_API_KEY")?;
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

fn text_content(role: &str, text: &str, content_type: ContentType) -> Value {
    let type_str = match content_type {
        ContentType::Input => "input_text",
        ContentType::Output => "output_text",
    };

    json!({
        "role": role,
        "content": [
            {
                "type": type_str,
                "text": text
            }
        ]
    })
}

enum ContentType {
    Input,
    Output,
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
