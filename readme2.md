# `Gibson` Repository Notes

`Gibson` currently contains three implementations of the same core idea: seeded procedural megastructure generation with interactive viewing.

## Repository Structure

* [README.md](/Users/gongahkia/Desktop/coding/projects/task9/gibson/README.md): Original project README. Left untouched on purpose.
* [Makefile](/Users/gongahkia/Desktop/coding/projects/task9/gibson/Makefile): Build and run entrypoint for the C++ implementation.
* [Cargo.toml](/Users/gongahkia/Desktop/coding/projects/task9/gibson/Cargo.toml): Cargo manifest for the Rust implementation.
* [src/main.py](/Users/gongahkia/Desktop/coding/projects/task9/gibson/src/main.py): Python generator and orbital-view visualizer using Pygame, ModernGL, and PyOpenGL.
* [src/main.cpp](/Users/gongahkia/Desktop/coding/projects/task9/gibson/src/main.cpp): C++ generator and viewer with richer generation phases, FPS traversal, collision, and post-processing.
* [src/main.rs](/Users/gongahkia/Desktop/coding/projects/task9/gibson/src/main.rs): Rust implementation that combines the C++ feature target with Python-style seed and JSON persistence.

## Purpose Of Each Implementation

### Python

The Python version is the lighter interactive renderer. It focuses on orbital navigation, seeded regeneration, structure serialization, screenshots, and inspection of generated cells.

### C++

The C++ version is the most ambitious renderer/generator. It expands generation with an L-system skeleton, WFC-based floor plans, spline infrastructure, erosion, FPS traversal, collision, and a richer post-processing pipeline.

### Rust

The Rust version is a hybrid port:

* Generation target: close to the C++ pipeline.
* Persistence target: close to the Python behavior.
* Viewer target: preserve orbital and FPS navigation, inspection, screenshots, fog/bloom controls, regeneration, and on-screen legend data.

## Rust Build And Run

```console
$ cargo run --release
$ cargo run --release -- ABCD1234
```

The optional command-line argument must be an 8-character alphanumeric seed.

## Rust Controls

* `Mouse Drag`: Rotate orbital camera
* `Mouse Wheel`: Zoom orbital camera
* `WASD`: Pan in orbital mode, move in FPS mode
* `1-5`: Camera presets
* `TAB`: Toggle FPS mode
* `Space`: Jump in FPS mode
* `R`: Regenerate with a new seed
* `S`: Save a screenshot
* `I`: Toggle inspection mode
* `P`: Toggle post-processing
* `[` and `]`: Decrease or increase fog density
* `-` and `=`: Decrease or increase bloom intensity
* `L`: Toggle legend
* `Q` or `Esc`: Quit

## Output Files

The Rust implementation writes outputs at the repository root:

* `current_seed.txt`: Current active seed
* `structure.json`: Serialized voxel structure
* `screenshots/`: Captured PNG screenshots

## Important Note

[README.md](/Users/gongahkia/Desktop/coding/projects/task9/gibson/README.md) was intentionally not edited during this port. This file exists to document the repository structure and the Rust implementation without changing the original project documentation.
