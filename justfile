check:
    cargo fmt --check
    cargo clippy -- -D warnings
    cargo test

fmt:
    cargo fmt

test:
    cargo test

build:
    cargo build --release

clean:
    cargo clean
