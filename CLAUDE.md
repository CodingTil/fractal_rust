# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GPU-accelerated fractal renderer written in Rust using WebGPU (wgpu). It generates animated Julia set fractals that smoothly zoom and transition between different interesting points in the complex plane. The project supports both native execution and WebAssembly for running in browsers.

## Build Commands

### Native Build
```bash
cargo build --release
```

### Run Native Binary
```bash
cargo run --release
```

### WebAssembly Build (with Trunk)
```bash
# Install trunk if not already installed
cargo install --locked trunk

# Add wasm32 target if not already added
rustup target add wasm32-unknown-unknown

# Build for web
trunk build --release

# Serve locally for testing
trunk serve
```

### Testing
```bash
cargo test --release --all-features
```

### Linting
```bash
# Run clippy with strict warnings
cargo clippy --all-features -- -D warnings

# Format check
cargo fmt --all -- --check

# Apply formatting
cargo fmt --all
```

## Architecture

### Entry Points
- **Native**: `src/main.rs` - Simple entry point that calls `pollster::block_on(run())`
- **WASM**: `src/lib.rs` - Exports `run()` and `run_with_canvas()` functions for web integration

### Core Structure (src/lib.rs)

The application follows a standard wgpu render loop pattern:

1. **State struct**: Central state manager holding GPU resources, uniforms, buffers, and timing information
2. **Uniform structs**: Two key uniform types passed to shaders:
   - `DimensionUniform` (src/lib.rs:26-49): Screen dimensions and inverse dimensions for coordinate mapping
   - `FractalUniform` (src/lib.rs:52-101): Fractal parameters including max iterations, complex point (c_real, c_imag), and zoom factor

3. **Platform-specific initialization**: State::new() has separate implementations for native and WASM (lines 162-214)
   - Native version takes a Window reference
   - WASM version takes an HtmlCanvasElement and creates the window from it

4. **Animation logic** (FractalUniform::update, lines 72-82):
   - Computes zoom using cosine-based easing: `0.5^(15.0 * (0.5 - 0.5 * cos(0.1275 * total_time)))`
   - Resets to a new random Julia set point when fully zoomed in and point has aged >10 seconds
   - Point selection from 9 predefined interesting locations (lines 86-96)

### Shader (src/shader.wgsl)

WGSL compute shader that:
- Takes vertex positions and converts to UV coordinates
- Iterates the Julia set formula: z_{n+1} = z_n^2 + c
- Implements smooth coloring based on iteration count
- Blends between Mandelbrot center (-0.5, 0) and target Julia point based on zoom_factor

### Render Pipeline
- Single fullscreen quad (2 triangles covering viewport)
- Vertex buffer contains 4 corners: [-1,-1], [-1,1], [1,-1], [1,1]
- Fragment shader computes per-pixel fractal iterations
- No depth buffer or multisampling

### Time Handling
- Cross-platform time: `std::time::SystemTime` for native, `web_time::SystemTime` for WASM
- Tracks total_time and current_point_age for animation and transition logic

## Key Design Patterns

- **Platform abstraction**: Uses `cfg` attributes extensively to separate native vs WASM code paths
- **Immutable vertex data**: VERTICES and INDICES are static constants
- **GPU-driven rendering**: All fractal computation happens in fragment shader for parallelism
- **Smooth transitions**: Zoom factor interpolates between two states, triggering point reset at extremes

## Known Issues

### EGL Sync Errors (Native Linux)
When running `cargo run` on Linux systems with certain GPU drivers, you may see EGL errors:
```
[ERROR wgpu_hal::gles::egl] EGL 'eglCreateSyncKHR' code 0x3004: EGL_BAD_ATTRIBUTE error
```

This is a known wgpu/driver issue with OpenGL backend fence synchronization. The program still runs correctly despite these errors being logged. These errors can typically be ignored as they don't affect rendering functionality.

## Dependencies

- `wgpu`: WebGPU API for cross-platform GPU access
- `winit`: Window creation and event handling
- `bytemuck`: Zero-copy casting for uniform buffers
- `pollster`: Async executor for native builds
- `wasm-bindgen`: WASM bindings for browser integration
- `rand`: Random point selection for fractal transitions
