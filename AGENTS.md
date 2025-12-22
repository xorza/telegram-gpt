AI coding rules for this project:
- Avoid using Option<> and Result<> for cases that should not fail.
- For required values, use `.expect("...")` with a clear, specific message.
- Prefer crashing on logic errors rather than silently swallowing them.
- Use Result<> only for expected/legitimate failures (e.g., network, I/O, external services).
- Always add `#[derive(Debug)]` to Rust structs.
- If Rust code was changed, run `cargo check` and `cargo clippy` before confirming output.
- Add asserts for function input arguments and outputs where applicable, so logic errors crash instead of being swallowed.
