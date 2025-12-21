#[derive(Debug)]
pub enum CommandArg {
    Empty,
    None,
    Text(String),
}

impl CommandArg {
    fn from_text(text: Option<&str>) -> Self {
        match text {
            Some(text) => {
                let trimmed = text.trim();
                if trimmed.is_empty() {
                    CommandArg::Empty
                } else if trimmed.eq_ignore_ascii_case("none") {
                    CommandArg::None
                } else {
                    CommandArg::Text(trimmed.to_string())
                }
            }
            None => CommandArg::Empty,
        }
    }
}

#[derive(Debug)]
pub enum Command {
    /// Ignore the current message.
    Ignore,
    /// Show this help text.
    Help,
    /// Show this help text.
    Start,
    /// List available models.
    Models,
    /// Get/set the model (use `none` to clear).
    Model(CommandArg),
    /// Get/set the API key (use `none` to clear).
    Key(CommandArg),
    /// Get/set the system prompt (use `none` to clear).
    SystemPrompt(CommandArg),
    /// List or update chat authorization.
    Approve(ApproveArg),
}

#[derive(Debug)]
pub enum ApproveArg {
    Empty,
    Invalid,
    ApproveChat { chat_id: i64, is_authorized: bool },
}

pub fn parse_command(text: &str, bot_username: &str) -> Result<Command, String> {
    let trimmed = text.trim();
    if !trimmed.starts_with('/') {
        return Err("Unknown command".to_string());
    }

    let without_slash = trimmed.trim_start_matches('/');
    let (cmd_part, args_part) = match without_slash.find(char::is_whitespace) {
        Some(idx) => (
            &without_slash[..idx],
            Some(without_slash[idx..].trim_start()),
        ),
        None => (without_slash, None),
    };
    let args_part = args_part.and_then(|args| if args.is_empty() { None } else { Some(args) });

    let (cmd_name, mention) = match cmd_part.split_once('@') {
        Some((cmd, mention)) => (cmd, Some(mention)),
        None => (cmd_part, None),
    };

    if let Some(mention) = mention
        && !mention.eq_ignore_ascii_case(bot_username)
    {
        return Ok(Command::Ignore);
    }

    match cmd_name.to_ascii_lowercase().as_str() {
        "help" => {
            if args_part.is_none() {
                Ok(Command::Help)
            } else {
                Err("Unknown command".to_string())
            }
        }
        "start" => {
            if args_part.is_none() {
                Ok(Command::Start)
            } else {
                Err("Unknown command".to_string())
            }
        }
        "models" => {
            if args_part.is_none() {
                Ok(Command::Models)
            } else {
                Err("Unknown command".to_string())
            }
        }
        "model" => Ok(Command::Model(CommandArg::from_text(args_part))),
        "key" => Ok(Command::Key(CommandArg::from_text(args_part))),
        "systemprompt" => Ok(Command::SystemPrompt(CommandArg::from_text(args_part))),
        "approve" => {
            if args_part.is_none() {
                return Ok(Command::Approve(ApproveArg::Empty));
            }
            let args = args_part.unwrap().split_whitespace().collect::<Vec<&str>>();
            if args.len() != 2 {
                return Ok(Command::Approve(ApproveArg::Invalid));
            }

            let chat_id: i64 = match args[0].parse() {
                Ok(value) => value,
                Err(_) => {
                    return Ok(Command::Approve(ApproveArg::Invalid));
                }
            };
            let is_authorized = match args[1].to_ascii_lowercase().as_str() {
                "true" | "1" => true,
                "false" | "0" => false,
                _ => {
                    return Ok(Command::Approve(ApproveArg::Invalid));
                }
            };
            Ok(Command::Approve(ApproveArg::ApproveChat {
                chat_id,
                is_authorized,
            }))
        }
        _ => Err("Unknown command".to_string()),
    }
}
