# VulkanScope Pro Made with Vulkan
<img width="260" height="100" alt="Vulkan_White_100px_Dec16" src="https://github.com/user-attachments/assets/f59ef502-3fb4-451e-b1ad-73f762001db8" />


<img width="1120" height="983" alt="66y" src="https://github.com/user-attachments/assets/ae885729-e16c-4988-b0df-53494c289004" />


Real-time Vulkan GPU-accelerated audio mastering & analysis suite for Linux (PipeWire + Wayland/X11).

## Implemented Features (v0.1 MVP)
- **PipeWire Real-Time Capture (`src/audio.rs`)**: Dedicated RT capture thread using `pipewire-rs` + lock-free `rtrb` SPSC ring buffer (zero allocations/locks in callback), live sink/app node switching, plus built-in mastering reference signals (Trance Kick + Stereo Synth, Lissajous Harmonics, 20Hz–20kHz Log Sweep).
- **8192-Point Vulkan Compute FFT (`src/shaders/fft.wgsl`)**: Full 13-stage Radix-2 GPU Compute FFT with Hann windowing, logarithmic frequency binning (20 Hz – 20 kHz), configurable pink-noise tilt (0 / 3 / 4.5 dB/oct), and direct writes into a `512 × 256` `R32Float` waterfall ring-buffer texture.
- **CRT Phosphor XY Oscilloscope (`src/shaders/phosphor.wgsl`)**: Velocity-modulated electron-beam capsule rendering into an HDR `Rgba16Float` persistence buffer with exponential phosphor decay, 45° Mid/Side mastering vectorscope rotation, etched CRT graticule, optional scanlines, and HDR bloom.
- **3D Spectrogram Waterfall (`src/shaders/waterfall.wgsl`)**: Real-time lit `256 × 192` 3D terrain mesh displaced from the GPU ring buffer with finite-difference normals, studio frequency markers (100 Hz / 1 kHz / 10 kHz), 4 palettes, interactive orbit camera, and HDR bloom (`src/shaders/composite.wgsl`).
- **Mastering Meters (`src/dsp.rs` & `src/main.rs`)**: EBU R128 Momentary & Short-Term LUFS, True Peak (`dBTP`), L/R Peak + RMS + Peak-Hold + Clip LED, Phase Correlation (`-1` to `+1`), and Stereo Width `%`.

## Build & Run

Build inside `my-ubuntu`:
```bash
podman exec my-ubuntu bash -lc 'export PATH="$HOME/.cargo/bin:$PATH"; cd /workspace/vulkanscope-pro && cargo build --release'
```

Run natively on the host (AMD RX 6500 XT Vulkan + PipeWire):
```bash
/home/shay/code1/vulkanscope-pro/target/release/vulkanscope-pro
```

## Keyboard & Mouse Controls
- **Left-Drag**: Orbit 3D Waterfall camera
- **Scroll Wheel**: Zoom 3D Waterfall camera
- **`1` / `2` / `3`**: Switch layout (`Split Studio` / `XY CRT` / `3D Waterfall`)
- **`Tab`**: Toggle floating Studio Controls inspector
- **`Space`**: Freeze / Unfreeze display
- **`H`**: Toggle Zen Mode (hide HUD)
- **`R`**: Reset 3D camera
- **`Esc`**: Quit
