use std::sync::Arc;

use tokio::sync::RwLock;

use crate::openrouter_api;

pub async fn spawn_model_refresh(
    http_client: reqwest::Client,
) -> Arc<RwLock<Vec<openrouter_api::ModelSummary>>> {
    let models = Arc::new(RwLock::new(Vec::new()));

    // Fetch helper keeps the refresh logic in one place.
    async fn refresh_models(
        http_client: &reqwest::Client,
        models: &Arc<RwLock<Vec<openrouter_api::ModelSummary>>>,
    ) -> anyhow::Result<()> {
        let latest = openrouter_api::list_models(http_client).await?;

        let mut guard = models.write().await;
        *guard = latest;

        Ok(())
    }

    // Run once immediately; keep retrying so we always start with a model list.
    let mut attempt = 1u32;
    loop {
        match refresh_models(&http_client, &models).await {
            Ok(()) => break,
            Err(err) => {
                log::warn!(
                    "initial model fetch failed (attempt {}): {err}; retrying in 5s",
                    attempt
                );
                attempt += 1;
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            }
        }
    }

    let models_clone = models.clone();
    let http_client = http_client.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10 * 60));
        loop {
            interval.tick().await;
            refresh_models(&http_client, &models_clone).await.ok();
        }
    });

    models
}
