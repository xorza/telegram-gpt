use flexi_logger::{Cleanup, Criterion, Duplicate, FileSpec, Logger, Naming};
use genai::{
    chat::{ChatMessage, ChatRequest},
    Client,
};
use teloxide::{prelude::*, types::Message};

type DynError = Box<dyn std::error::Error + Send + Sync>;

const DEFAULT_MODEL: &str = "gpt-4o-mini";

#[tokio::main]
async fn main() -> Result<(), DynError> {
    dotenv::dotenv().ok();
    // Log to rotating files capped at 10MB each, keeping the 3 newest, while also duplicating info logs to stdout.
    Logger::try_with_env_or_str("info")?
        .log_to_file(FileSpec::default().directory("logs"))
        .rotate(
            Criterion::Size(10 * 1024 * 1024),
            Naming::Numbers,
            Cleanup::KeepLogFiles(3),
        )
        .duplicate_to_stdout(Duplicate::Warn)
        .start()?;
    log::info!("starting tggpt bot");

    let bot = Bot::from_env();
    let client = Client::default();
    let model = std::env::var("GENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let system_prompt = std::env::var("GENAI_SYSTEM_PROMPT").ok();

    teloxide::repl(bot, move |bot: Bot, msg: Message| {
        let client = client.clone();
        let model = model.clone();
        let system_prompt = system_prompt.clone();
        async move {
            if let Some(user_text) = msg.text() {
                match send_to_llm(&client, &model, system_prompt.as_deref(), user_text).await {
                    Ok(answer) => {
                        bot.send_message(msg.chat.id, answer).await?;
                    }
                    Err(err) => {
                        log::error!("failed to get llm response: {err}");
                        bot.send_message(
                            msg.chat.id,
                            "I couldn't reach the language model. Please try again.",
                        )
                        .await?;
                    }
                }
            } else {
                bot.send_message(
                    msg.chat.id,
                    "Please send text messages so I can ask the language model.",
                )
                .await?;
            }
            respond(())
        }
    })
    .await;

    Ok(())
}

async fn send_to_llm(
    client: &Client,
    model: &str,
    system_prompt: Option<&str>,
    user_text: &str,
) -> Result<String, DynError> {
    let mut messages = Vec::with_capacity(2);
    if let Some(prompt) = system_prompt {
        messages.push(ChatMessage::system(prompt));
    }
    messages.push(ChatMessage::user(user_text));

    let chat_req = ChatRequest::new(messages);
    let chat_res = client.exec_chat(model, chat_req, None).await?;

    let answer = chat_res
        .first_text()
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "The language model returned an empty response.".to_string());

    Ok(answer)
}
