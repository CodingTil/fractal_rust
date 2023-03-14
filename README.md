# Fractal Rust (GPU)

![Fractal Image](./fractal.webp)

## Build
### Native
```bash
cargo build --release
```

### WASM
```bash
wasm-pack build --release
```

### WASM for Web
```bash
trunk build --release
# or
wasm-pack build --release --target web
```