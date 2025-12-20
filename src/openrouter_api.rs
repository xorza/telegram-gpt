use crate::conversation::{Message, MessageRole};
use anyhow::{Context, anyhow};
use log::info;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

#[allow(dead_code)]
const MODELS_ENDPOINT: &str = "https://openrouter.ai/api/v1/models";

#[derive(Debug)]
enum ContentType {
    Input,
    Output,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelSummary {
    pub id: String,
    pub name: String,
    pub context_length: u64,
    /// USD cost per million prompt token.
    pub prompt_m_token_cost_usd: f64,
    /// USD cost per million completion/output token.
    pub completion_m_token_cost_usd: f64,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelRecord>,
}

#[derive(Debug, Deserialize)]
struct ModelRecord {
    id: String,
    name: String,
    context_length: u64,
    pricing: Pricing,
}

#[derive(Debug, Deserialize, Default)]
struct Pricing {
    prompt: String,
    completion: String,
}

#[derive(Debug)]
pub struct Response {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cost: f64,
    pub completion_text: String,
}

pub fn estimate_tokens<'a, I>(messages: I) -> u64
where
    I: IntoIterator<Item = &'a str>,
{
    const AVG_BYTES_PER_TOKEN: u64 = 4;
    const PER_MESSAGE_OVERHEAD: u64 = 10;

    let (byte_count, message_count) = messages
        .into_iter()
        .fold((0u64, 0u64), |(bytes, msgs), message| {
            (bytes + message.len() as u64, msgs + 1)
        });

    let text_tokens = byte_count.div_ceil(AVG_BYTES_PER_TOKEN);

    let tokens = text_tokens + message_count * PER_MESSAGE_OVERHEAD;

    // reserve 1/4 of tokens for completion
    tokens * 5 / 4
}

pub async fn list_models(http: &Client) -> anyhow::Result<Vec<ModelSummary>> {
    let request = http.get(MODELS_ENDPOINT);

    let response = request
        .send()
        .await
        .context("failed to query OpenRouter models")?;

    let status = response.status();
    let body = response.text().await?;
    if !status.is_success() {
        return Err(anyhow::anyhow!(
            "OpenRouter models endpoint returned {status}: {body}"
        ));
    }

    let parsed: ModelsResponse =
        serde_json::from_str(&body).context("failed to parse OpenRouter models response JSON")?;

    Ok(parsed.data.into_iter().map(model_to_summary).collect())
}

pub async fn model(http: &Client, model: &str) -> anyhow::Result<ModelSummary> {
    let models = list_models(http).await?;
    let model = models
        .into_iter()
        .find(|m| m.id == model)
        .ok_or_else(|| anyhow::anyhow!("model not found"))?;

    Ok(model)
}

#[allow(dead_code)]
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
        "plugins": [
            { "id": "web" }
        ],
        "usage": { "include": true },
        "stream": stream,
    })
}

#[allow(dead_code)]
pub async fn send<F, Fut>(
    http: &Client,
    api_key: &str,
    payload: serde_json::Value,
    stream: bool,
    on_delta: F,
) -> anyhow::Result<Response>
where
    F: FnMut(String, bool) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>> + Send,
{
    if stream {
        send_streaming(http, api_key, payload, on_delta).await
    } else {
        send_no_streaming(http, api_key, payload, on_delta).await
    }
}

async fn send_streaming<F, Fut>(
    _http: &Client,
    _api_key: &str,
    _payload: serde_json::Value,
    mut _on_delta: F,
) -> anyhow::Result<Response>
where
    F: FnMut(String, bool) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>> + Send,
{
    unimplemented!()
}

async fn send_no_streaming<F, Fut>(
    http: &Client,
    api_key: &str,
    payload: serde_json::Value,
    mut on_delta: F,
) -> anyhow::Result<Response>
where
    F: FnMut(String, bool) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>> + Send,
{
    let response = http
        .post("https://openrouter.ai/api/v1/responses")
        .bearer_auth(api_key)
        .json(&payload)
        .send()
        .await?;

    let status = response.status();
    let body_text = response.text().await?;

    if !status.is_success() {
        return Err(anyhow!(
            "OpenRouter Responses API error {status}: {body_text}"
        ));
    }

    let response_body: serde_json::Value = serde_json::from_str(&body_text)?;

    let response = extract_output_text(&response_body);
    if !response.completion_text.is_empty() {
        on_delta(response.completion_text.clone(), true).await?;
        return Ok(response);
    }

    Err(anyhow!(
        "OpenRouter response missing text output: {response_body}"
    ))
}

fn extract_output_text(value: &serde_json::Value) -> Response {
    let text = value
        .get("output")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|v| v.get("content").and_then(|c| c.as_array()))
        .flatten()
        .filter_map(|v| v.get("text").and_then(|t| t.as_str()))
        .collect::<Vec<&str>>()
        .join("\n")
        .trim()
        .to_string();

    let usage = value.get("usage").expect("Missing usage");

    Response {
        prompt_tokens: usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .expect("Missing input_tokens"),
        completion_tokens: usage
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .expect("Missing output_tokens"),
        total_tokens: usage
            .get("total_tokens")
            .and_then(|v| v.as_u64())
            .expect("Missing total_tokens"),
        cost: usage
            .get("cost")
            .and_then(|v| v.as_f64())
            .expect("Missing cost"),
        completion_text: text,
    }
}

fn model_to_summary(model: ModelRecord) -> ModelSummary {
    let pricing = model.pricing;

    ModelSummary {
        id: model.id,
        name: model.name,
        context_length: model.context_length,
        prompt_m_token_cost_usd: 1_000_000.0 * pricing.prompt.parse::<f64>().unwrap(),
        completion_m_token_cost_usd: 1_000_000.0 * pricing.completion.parse::<f64>().unwrap(),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sample_payload() {
        let payload = r#"
        {
          "data": [
            {
              "id": "openai/gpt-4",
              "name": "GPT-4",
              "context_length": 8192,
              "pricing": {
                "prompt": "0.00003",
                "completion": "0.00006"
              }
            }
          ]
        }"#;

        let parsed: ModelsResponse = serde_json::from_str(payload).unwrap();
        let summaries: Vec<ModelSummary> = parsed.data.into_iter().map(model_to_summary).collect();

        assert_eq!(summaries.len(), 1);
        let model = &summaries[0];
        assert_eq!(model.id, "openai/gpt-4");
        assert_eq!(model.name.as_str(), "GPT-4");
        assert_eq!(model.context_length, 8192);
        assert_eq!(model.prompt_m_token_cost_usd, 1_000_000.0 * 0.00003);
        assert_eq!(model.completion_m_token_cost_usd, 1_000_000.0 * 0.00006);
    }

    // Integration test that calls the live OpenRouter models endpoint.
    #[tokio::test(flavor = "multi_thread")]
    async fn live_openrouter_models() {
        let http = reqwest::Client::new();
        let models = list_models(&http).await.expect("live models fetch failed");

        assert!(
            !models.is_empty(),
            "OpenRouter returned no models in live fetch"
        );
        assert!(
            models.iter().any(|m| !m.id.is_empty()),
            "expected at least one model id"
        );
    }

    // Integration test that calls the live OpenRouter responses endpoint (non-streaming).
    #[tokio::test(flavor = "multi_thread")]
    async fn live_send_no_streaming_returns_text() {
        let http = reqwest::Client::new();
        let api_key =
            std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY env var not set");
        let model = std::env::var("OPENROUTER_TEST_MODEL")
            .unwrap_or_else(|_| "xiaomi/mimo-v2-flash:free".to_string());

        let user_message = Message {
            role: MessageRole::User,
            text: "Say hello in one short sentence.".to_string(),
        };

        let payload = prepare_payload(&model, std::iter::once(&user_message), false);

        let buffer = std::sync::Arc::new(tokio::sync::Mutex::new(String::new()));
        let buffer_clone = buffer.clone();

        let result = send_no_streaming(
            &http,
            &api_key,
            payload,
            move |delta: String, finalize: bool| {
                let buffer = buffer_clone.clone();
                async move {
                    let mut buf = buffer.lock().await;
                    buf.push_str(&delta);
                    if finalize {
                        println!("Model reponse: {}", buf);
                    }
                    Ok(())
                }
            },
        )
        .await
        .expect("send_no_streaming failed");

        let captured = buffer.lock().await.clone();
        assert!(
            result.completion_tokens > 0,
            "LLM response should not be empty"
        );
        assert!(
            !captured.trim().is_empty(),
            "on_delta should have received content"
        );
    }
}
