use std::panic::PanicHookInfo;

pub fn fatal_panic(message: impl std::fmt::Display) -> ! {
    log::error!("fatal: {}", message);
    std::process::exit(1);
}

pub fn set_panic_hook() {
    std::panic::set_hook(Box::new(|info: &PanicHookInfo| {
        let payload = if let Some(msg) = info.payload().downcast_ref::<&str>() {
            (*msg).to_string()
        } else if let Some(msg) = info.payload().downcast_ref::<String>() {
            msg.clone()
        } else {
            "panic occurred".to_string()
        };

        let location = info
            .location()
            .map(|loc| format!(" at {}:{}", loc.file(), loc.line()));

        if let Some(location) = location {
            fatal_panic(format!("{}{}", payload, location));
        } else {
            fatal_panic(payload);
        }
    }));
}
