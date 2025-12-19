use crate::conversation::{Message, MessageRole};
use anyhow::Context;
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
    pub context_length: usize,
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
    context_length: usize,
    pricing: Pricing,
}

#[derive(Debug, Deserialize, Default)]
struct Pricing {
    prompt: String,
    completion: String,
}

/// Fetch available OpenRouter models, returning their ids, context limits, and token prices.
#[allow(dead_code)]
pub async fn list_models(http: &Client, api_key: &str) -> anyhow::Result<Vec<ModelSummary>> {
    let request = http.get(MODELS_ENDPOINT).bearer_auth(api_key);

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
    on_delta: F,
) -> anyhow::Result<String>
where
    F: FnMut(String, bool) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>> + Send,
{
    // if stream {
    // send_streaming(http, api_key, payload, on_delta).await
    // } else {
    send_no_streaming(http, api_key, payload, on_delta).await
    // }
}

async fn send_no_streaming<F, Fut>(
    http: &Client,
    api_key: &str,
    payload: serde_json::Value,
    mut on_delta: F,
) -> anyhow::Result<String>
where
    F: FnMut(String, bool) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>> + Send,
{
    Ok("".to_string())
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
        let api_key =
            std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY env var not set");
        let models = list_models(&http, &api_key)
            .await
            .expect("live models fetch failed");

        assert!(
            !models.is_empty(),
            "OpenRouter returned no models in live fetch"
        );
        assert!(
            models.iter().any(|m| !m.id.is_empty()),
            "expected at least one model id"
        );
    }
}
