mod audio;
mod dsp;
mod renderer;

use audio::{AudioCapture, AudioSource, TestSignalMode};
use dsp::DspProcessor;
use egui::{Color32, CornerRadius, Margin, Rect, RichText, Stroke, Vec2};
use renderer::{LayoutMode, PhosphorColor, ScopeSettings, VulkanRenderer, WaterfallPalette};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
#[cfg(target_os = "linux")]
use winit::platform::wayland::WindowAttributesExtWayland;
use winit::window::{Icon, Window, WindowId, WindowLevel};

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<VulkanRenderer>,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,

    audio: AudioCapture,
    dsp: DspProcessor,
    drain_buf: Vec<[f32; 2]>,
    settings: ScopeSettings,

    // UI & Interaction state
    show_hud: bool,
    show_inspector: bool,
    always_on_top: bool,
    dragging_camera: bool,
    last_cursor: Option<(f64, f64)>,

    // Timing / FPS
    last_frame: Instant,
    fps_smooth: f32,
}

impl App {
    fn new() -> Self {
        let audio = AudioCapture::new();
        let sr = audio.sample_rate();
        let egui_ctx = egui::Context::default();
        configure_studio_theme(&egui_ctx);

        Self {
            window: None,
            renderer: None,
            egui_ctx,
            egui_state: None,
            audio,
            dsp: DspProcessor::new(sr),
            drain_buf: Vec::with_capacity(8192),
            settings: ScopeSettings::default(),
            show_hud: true,
            show_inspector: false,
            always_on_top: false,
            dragging_camera: false,
            last_cursor: None,
            last_frame: Instant::now(),
            fps_smooth: 60.0,
        }
    }

    fn build_ui(&mut self, root_ui: &mut egui::Ui, adapter_name: &str, backend_name: &str) -> (f32, f32) {
        if !self.show_hud {
            // Subtle floating hint in Zen Mode
            egui::Area::new(egui::Id::unique("zen_hint"))
                .anchor(egui::Align2::RIGHT_TOP, [-12.0, 10.0])
                .show(&self.egui_ctx, |ui| {
                    ui.label(
                        RichText::new(format!("{:.0} FPS · Press H for HUD", self.fps_smooth))
                            .size(11.0)
                            .color(Color32::from_rgba_unmultiplied(150, 210, 225, 110)),
                    );
                });
            return (0.0, 0.0);
        }

        let screen_rect = self.egui_ctx.viewport_rect();
        let win_w = screen_rect.width().max(720.0);
        let top_h = 40.0;
        let bot_h = 86.0;

        let mut toggle_on_top = false;

        // ---- TOP STUDIO BAR ----
        egui::Panel::top("top_studio_bar")
            .exact_size(top_h)
            .frame(
                egui::Frame::NONE
                    .fill(Color32::from_rgb(12, 16, 24))
                    .inner_margin(Margin::symmetric(12, 6))
                    .stroke(Stroke::new(1.0, Color32::from_rgb(28, 42, 58))),
            )
            .show(root_ui, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.label(
                        RichText::new("VULKANSCOPE")
                            .strong()
                            .size(14.0)
                            .color(Color32::from_rgb(56, 240, 222)),
                    );
                    ui.label(
                        RichText::new("PRO")
                            .strong()
                            .size(10.5)
                            .color(Color32::from_rgb(255, 175, 65)),
                    );
                    ui.separator();

                    // Live status dot
                    let (status_col, status_txt) = match self.audio.source {
                        AudioSource::TestSignal(_) => {
                            (Color32::from_rgb(255, 185, 60), "TEST")
                        }
                        _ if self.audio.is_pipewire_connected() => {
                            if self.dsp.metrics.has_signal {
                                (Color32::from_rgb(48, 242, 145), "LIVE")
                            } else {
                                (Color32::from_rgb(80, 185, 225), "IDLE")
                            }
                        }
                        _ => (Color32::from_rgb(235, 85, 85), "NO PW"),
                    };
                    let (dot_rect, _) = ui.allocate_exact_size(Vec2::splat(8.0), egui::Sense::hover());
                    ui.painter().circle_filled(dot_rect.center(), 4.0, status_col);
                    ui.label(RichText::new(status_txt).size(11.0).strong().color(status_col));

                    // PipeWire / Test Signal Source Dropdown (responsive width)
                    let nodes = self.audio.available_nodes();
                    let current_label = self.audio.source.display_name();
                    let mut next_source: Option<AudioSource> = None;
                    let combo_w = if win_w < 1160.0 { 185.0 } else { 235.0 };

                    egui::ComboBox::from_id_salt("audio_source_combo")
                        .width(combo_w)
                        .selected_text(RichText::new(current_label).size(11.5))
                        .show_ui(ui, |ui| {
                            if ui
                                .selectable_label(
                                    self.audio.source == AudioSource::DefaultMonitor,
                                    "PipeWire: System Default Output (Monitor)",
                                )
                                .clicked()
                            {
                                next_source = Some(AudioSource::DefaultMonitor);
                            }

                            if !nodes.is_empty() {
                                ui.separator();
                                ui.label(
                                    RichText::new("PipeWire Nodes")
                                        .size(10.5)
                                        .color(Color32::from_rgb(130, 160, 185)),
                                );
                                for node in &nodes {
                                    let candidate = AudioSource::PipeWireNode {
                                        name: node.name.clone(),
                                        label: node.label.clone(),
                                    };
                                    if ui
                                        .selectable_label(self.audio.source == candidate, &node.label)
                                        .clicked()
                                    {
                                        next_source = Some(candidate);
                                    }
                                }
                            }

                            ui.separator();
                            ui.label(
                                RichText::new("Built-in Reference Signals")
                                    .size(10.5)
                                    .color(Color32::from_rgb(245, 180, 75)),
                            );
                            for mode in [
                                TestSignalMode::TranceKickSynth,
                                TestSignalMode::LissajousHarmonic,
                                TestSignalMode::FreqSweep,
                            ] {
                                let candidate = AudioSource::TestSignal(mode);
                                if ui
                                    .selectable_label(self.audio.source == candidate, mode.label())
                                    .clicked()
                                {
                                    next_source = Some(candidate);
                                }
                            }
                        });

                    if let Some(src) = next_source {
                        self.audio.set_source(src);
                    }

                    ui.separator();

                    // Layout Switcher
                    ui.selectable_value(&mut self.settings.layout, LayoutMode::SplitStudio, "Split");
                    ui.selectable_value(&mut self.settings.layout, LayoutMode::PhosphorOnly, "XY CRT");
                    ui.selectable_value(&mut self.settings.layout, LayoutMode::WaterfallOnly, "3D Waterfall");

                    ui.separator();

                    // Freeze, On Top & Controls buttons
                    let freeze_btn = ui.selectable_label(
                        self.settings.frozen,
                        RichText::new(if self.settings.frozen { "FROZEN" } else { "Freeze" }).size(11.5),
                    );
                    if freeze_btn.clicked() {
                        self.settings.frozen = !self.settings.frozen;
                    }

                    if ui
                        .selectable_label(self.always_on_top, RichText::new("Pin").size(11.5))
                        .on_hover_text("Always on top (floating mastering window)")
                        .clicked()
                    {
                        toggle_on_top = true;
                    }

                    if ui
                        .selectable_label(self.show_inspector, RichText::new("Controls").size(11.5))
                        .clicked()
                    {
                        self.show_inspector = !self.show_inspector;
                    }

                    // Right-aligned telemetry — only draw GPU label if there is clean room (zero overlap!)
                    let rem_w = ui.available_width();
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("{:.0} FPS", self.fps_smooth))
                                .monospace()
                                .strong()
                                .size(12.0)
                                .color(Color32::from_rgb(72, 245, 195)),
                        );
                        if rem_w > 240.0 {
                            ui.label(
                                RichText::new(format!(
                                    "{} · {} · {}kHz",
                                    short_gpu_name(adapter_name),
                                    backend_name,
                                    self.audio.sample_rate() / 1000
                                ))
                                .size(10.5)
                                .color(Color32::from_rgb(125, 148, 172)),
                            );
                        } else if rem_w > 110.0 {
                            ui.label(
                                RichText::new(format!("{}kHz", self.audio.sample_rate() / 1000))
                                    .size(10.5)
                                    .color(Color32::from_rgb(125, 148, 172)),
                            );
                        }
                    });
                });
            });

        if toggle_on_top {
            self.always_on_top = !self.always_on_top;
            if let Some(w) = &self.window {
                w.set_window_level(if self.always_on_top {
                    WindowLevel::AlwaysOnTop
                } else {
                    WindowLevel::Normal
                });
            }
        }

        // ---- BOTTOM MASTERING METER BAR ----
        let metrics = self.dsp.metrics;
        let mut reset_meters = false;
        let meter_w = if win_w < 1100.0 {
            175.0
        } else if win_w < 1320.0 {
            215.0
        } else {
            265.0
        };
        let corr_w = if win_w < 1100.0 { 145.0 } else { 185.0 };

        egui::Panel::bottom("bottom_mastering_bar")
            .exact_size(bot_h)
            .frame(
                egui::Frame::NONE
                    .fill(Color32::from_rgb(11, 14, 21))
                    .inner_margin(Margin::symmetric(14, 10))
                    .stroke(Stroke::new(1.0, Color32::from_rgb(28, 42, 58))),
            )
            .show(root_ui, |ui| {
                ui.horizontal_centered(|ui| {
                    // 1. Stereo L/R Peak + RMS Meters
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("OUTPUT LEVEL (dBFS)")
                                    .size(10.0)
                                    .strong()
                                    .color(Color32::from_rgb(135, 165, 190)),
                            );
                            if metrics.clipped {
                                ui.label(
                                    RichText::new(" CLIP ")
                                        .size(10.0)
                                        .strong()
                                        .background_color(Color32::from_rgb(210, 35, 45))
                                        .color(Color32::WHITE),
                                );
                            }
                        });
                        ui.add_space(4.0);
                        draw_level_meter(ui, "L", metrics.peak_l_db, metrics.rms_l_db, metrics.hold_l_db, meter_w);
                        ui.add_space(4.0);
                        draw_level_meter(ui, "R", metrics.peak_r_db, metrics.rms_r_db, metrics.hold_r_db, meter_w);
                    });

                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // 2. Phase Correlation & Stereo Width
                    ui.vertical(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("PHASE CORRELATION")
                                    .size(10.0)
                                    .strong()
                                    .color(Color32::from_rgb(135, 165, 190)),
                            );
                            ui.add_space(6.0);
                            let corr_col = if metrics.correlation < 0.0 {
                                Color32::from_rgb(255, 85, 85)
                            } else if metrics.correlation < 0.35 {
                                Color32::from_rgb(255, 195, 65)
                            } else {
                                Color32::from_rgb(56, 240, 210)
                            };
                            ui.label(
                                RichText::new(format!("{:+.2}", metrics.correlation))
                                    .monospace()
                                    .strong()
                                    .size(11.5)
                                    .color(corr_col),
                            );
                        });
                        ui.add_space(5.0);
                        draw_correlation_bar(ui, metrics.correlation, corr_w);
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new(format!(
                                "Width: {:3.0}% · {}",
                                metrics.stereo_width * 100.0,
                                if self.settings.rotate_ms { "M/S 45°" } else { "XY L/R" }
                            ))
                            .monospace()
                            .size(10.5)
                            .color(Color32::from_rgb(145, 175, 195)),
                        );
                    });

                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // 3. EBU R128 LUFS & True Peak Readouts
                    draw_readout_card(
                        ui,
                        "LUFS-M",
                        format_db(metrics.lufs_momentary, -69.5),
                        Color32::from_rgb(68, 236, 224),
                    );
                    ui.add_space(6.0);
                    draw_readout_card(
                        ui,
                        "LUFS-S",
                        format_db(metrics.lufs_short_term, -69.5),
                        Color32::from_rgb(120, 245, 165),
                    );
                    ui.add_space(6.0);
                    let tp_col = if metrics.true_peak_db > -1.0 {
                        Color32::from_rgb(255, 92, 92)
                    } else {
                        Color32::from_rgb(255, 200, 90)
                    };
                    draw_readout_card(
                        ui,
                        "TRUE PEAK",
                        format!("{} dBTP", format_db(metrics.true_peak_db, -85.0)),
                        tp_col,
                    );

                    let rem_bot_w = ui.available_width();
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Reset").clicked() {
                            reset_meters = true;
                        }
                        if rem_bot_w > 325.0 {
                            ui.label(
                                RichText::new("Drag: Orbit · Scroll: Zoom · H: Zen · Space: Freeze")
                                    .size(10.5)
                                    .color(Color32::from_rgb(95, 118, 140)),
                            );
                        }
                    });
                });
            });

        if reset_meters {
            self.dsp.reset_meters();
        }

        // ---- SUBTLE CANVAS OVERLAY LABELS (FREQUENCY AXIS + CRT MARKERS) ----
        draw_canvas_annotations(
            &self.egui_ctx,
            screen_rect,
            top_h,
            bot_h,
            &self.settings,
            self.audio.sample_rate(),
        );

        // ---- COLLAPSIBLE STUDIO INSPECTOR ----
        if self.show_inspector {
            egui::Window::new("Studio Controls")
                .anchor(egui::Align2::RIGHT_TOP, [-12.0, top_h + 10.0])
                .resizable(false)
                .collapsible(false)
                .max_width(275.0)
                .frame(
                    egui::Frame::window(&self.egui_ctx.style_of(egui::Theme::Dark))
                        .fill(Color32::from_rgba_unmultiplied(13, 18, 27, 236))
                        .stroke(Stroke::new(1.0, Color32::from_rgb(42, 68, 92))),
                )
                .show(&self.egui_ctx, |ui| {
                    ui.label(
                        RichText::new("CRT PHOSPHOR XY SCOPE")
                            .size(11.0)
                            .strong()
                            .color(Color32::from_rgb(65, 238, 220)),
                    );
                    ui.add_space(4.0);

                    ui.horizontal(|ui| {
                        ui.label("Phosphor:");
                        egui::ComboBox::from_id_salt("phosphor_preset")
                            .selected_text(self.settings.phosphor_color.label())
                            .show_ui(ui, |ui| {
                                for p in [
                                    PhosphorColor::TektronixCyan,
                                    PhosphorColor::LabGreen,
                                    PhosphorColor::AmberCrt,
                                    PhosphorColor::PlasmaViolet,
                                ] {
                                    ui.selectable_value(&mut self.settings.phosphor_color, p, p.label());
                                }
                            });
                    });

                    ui.checkbox(&mut self.settings.rotate_ms, "45° Mid/Side Mastering Rotation");
                    ui.add(
                        egui::Slider::new(&mut self.settings.decay_tau, 0.012..=0.22)
                            .text("Persistence (s)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.settings.beam_intensity, 0.15..=1.6)
                            .text("Beam Drive"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.settings.beam_radius, 0.005..=0.024)
                            .text("Beam Focus"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.settings.scope_gain, 0.5..=2.5)
                            .text("Input Gain"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.settings.scanlines, 0.0..=1.0)
                            .text("CRT Scanlines"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.settings.graticule_alpha, 0.0..=1.2)
                            .text("Graticule Grid"),
                    );

                    ui.separator();
                    ui.label(
                        RichText::new("8192-PT FFT & 3D WATERFALL")
                            .size(11.0)
                            .strong()
                            .color(Color32::from_rgb(120, 230, 255)),
                    );
                    ui.add_space(4.0);

                    ui.horizontal(|ui| {
                        ui.label("Palette:");
                        egui::ComboBox::from_id_salt("wf_palette")
                            .selected_text(self.settings.palette.label())
                            .show_ui(ui, |ui| {
                                for p in [
                                    WaterfallPalette::VulkanCyber,
                                    WaterfallPalette::MagmaStudio,
                                    WaterfallPalette::CrtPhosphor,
                                    WaterfallPalette::IceLab,
                                ] {
                                    ui.selectable_value(&mut self.settings.palette, p, p.label());
                                }
                            });
                    });

                    ui.checkbox(&mut self.settings.flow_toward, "Flow Toward Camera");
                    ui.horizontal(|ui| {
                        ui.label("Tilt:");
                        for (val, txt) in [(0.0, "0 dB"), (3.0, "3 dB/oct"), (4.5, "4.5 dB/oct")] {
                            let active = (self.settings.tilt_db_per_oct - val).abs() < 0.1;
                            if ui.selectable_label(active, txt).clicked() {
                                self.settings.tilt_db_per_oct = val;
                            }
                        }
                    });
                    ui.add(
                        egui::Slider::new(&mut self.settings.smoothing, 0.0..=0.90)
                            .text("FFT Smoothing"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.settings.height_scale, 0.8..=4.5)
                            .text("Mountain Height"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.settings.wireframe_mix, 0.0..=1.0)
                            .text("Contour Grid"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.settings.glow_boost, 0.0..=2.5)
                            .text("Peak Glow"),
                    );

                    ui.separator();
                    ui.label(
                        RichText::new("VULKAN HDR POST-PROCESS")
                            .size(11.0)
                            .strong()
                            .color(Color32::from_rgb(255, 190, 95)),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.settings.bloom_strength, 0.0..=2.0)
                            .text("HDR Bloom"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.settings.exposure, 0.6..=1.8)
                            .text("Exposure"),
                    );
                    if self.settings.layout == LayoutMode::SplitStudio {
                        ui.add(
                            egui::Slider::new(&mut self.settings.split_ratio, 0.25..=0.60)
                                .text("Split Ratio"),
                        );
                    }

                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button("Reset Camera (R)").clicked() {
                            self.settings.reset_camera();
                        }
                        if ui.button("Defaults").clicked() {
                            self.settings = ScopeSettings::default();
                        }
                    });
                });
        }

        (top_h, bot_h)
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let mut attrs = Window::default_attributes()
            .with_title("VulkanScope Pro — PipeWire Mastering Analyzer")
            .with_inner_size(LogicalSize::new(1480.0, 820.0))
            .with_min_inner_size(LogicalSize::new(820.0, 460.0))
            .with_window_icon(build_window_icon());

        #[cfg(target_os = "linux")]
        {
            attrs = attrs.with_name("io.github.vulkanscope.VulkanScopePro", "VulkanScopePro");
        }

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        let renderer = VulkanRenderer::new(Arc::clone(&window));
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.egui_state = Some(egui_state);
        self.last_frame = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.clone() else {
            return;
        };

        let egui_consumed = if let Some(state) = &mut self.egui_state {
            state.on_window_event(&window, &event).consumed
        } else {
            false
        };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(new_size.width, new_size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                if !self.egui_ctx.egui_wants_keyboard_input() {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        match code {
                            KeyCode::Space => self.settings.frozen = !self.settings.frozen,
                            KeyCode::Tab => self.show_inspector = !self.show_inspector,
                            KeyCode::KeyH => self.show_hud = !self.show_hud,
                            KeyCode::KeyR => self.settings.reset_camera(),
                            KeyCode::Digit1 => self.settings.layout = LayoutMode::SplitStudio,
                            KeyCode::Digit2 => self.settings.layout = LayoutMode::PhosphorOnly,
                            KeyCode::Digit3 => self.settings.layout = LayoutMode::WaterfallOnly,
                            KeyCode::Escape => event_loop.exit(),
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                if state == ElementState::Pressed && !egui_consumed && !self.egui_ctx.is_pointer_over_egui() {
                    self.dragging_camera = true;
                } else if state == ElementState::Released {
                    self.dragging_camera = false;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some((lx, ly)) = self.last_cursor {
                    if self.dragging_camera {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        self.settings.cam_yaw -= dx * 0.0065;
                        self.settings.cam_pitch =
                            (self.settings.cam_pitch + dy * 0.0055).clamp(0.14, 1.45);
                    }
                }
                self.last_cursor = Some((position.x, position.y));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if !egui_consumed && !self.egui_ctx.is_pointer_over_egui() {
                    let scroll_y = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => (pos.y as f32) * 0.03,
                    };
                    self.settings.cam_distance =
                        (self.settings.cam_distance - scroll_y * 0.55).clamp(4.2, 16.5);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last_frame).as_secs_f32().clamp(0.0005, 0.1);
                self.last_frame = now;
                let inst_fps = 1.0 / dt;
                self.fps_smooth += (inst_fps - self.fps_smooth) * (dt * 6.0).clamp(0.02, 0.25);

                // Drain audio samples & update mastering DSP
                let sr = self.audio.sample_rate();
                self.audio.drain_samples(dt, &mut self.drain_buf);
                if !self.settings.frozen {
                    self.dsp.push_samples(&self.drain_buf, sr, dt);
                }

                // Run egui frame
                let raw_input = self
                    .egui_state
                    .as_mut()
                    .unwrap()
                    .take_egui_input(&window);

                let (adapter_name, backend_name) = {
                    let r = self.renderer.as_ref().unwrap();
                    (r.adapter_name.clone(), r.backend_name.clone())
                };

                let mut top_logical = 0.0;
                let mut bot_logical = 0.0;
                let egui_ctx = self.egui_ctx.clone();
                let egui_output = egui_ctx.run_ui(raw_input, |root_ui| {
                    let (t, b) = self.build_ui(root_ui, &adapter_name, &backend_name);
                    top_logical = t;
                    bot_logical = b;
                });

                self.egui_state
                    .as_mut()
                    .unwrap()
                    .handle_platform_output(&window, egui_output.platform_output.clone());

                let ppp = egui_output.pixels_per_point;
                let top_px = top_logical * ppp;
                let bot_px = bot_logical * ppp;

                if let Some(r) = &mut self.renderer {
                    r.render_frame(
                        &window,
                        self.dsp.window(),
                        self.dsp.last_fresh_count,
                        sr,
                        dt,
                        &self.settings,
                        top_px,
                        bot_px,
                        &self.egui_ctx,
                        egui_output,
                        ppp,
                    );
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn configure_studio_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = Color32::from_rgb(14, 19, 28);
    visuals.panel_fill = Color32::from_rgb(12, 16, 24);
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(24, 34, 48);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(34, 52, 72);
    visuals.widgets.active.bg_fill = Color32::from_rgb(44, 78, 102);
    visuals.selection.bg_fill = Color32::from_rgb(24, 130, 145);
    visuals.selection.stroke = Stroke::new(1.0, Color32::from_rgb(72, 245, 225));
    ctx.set_visuals(visuals);
}

fn short_gpu_name(name: &str) -> String {
    if name.len() > 28 {
        format!("{}…", &name[..27])
    } else {
        name.to_string()
    }
}

fn format_db(val: f32, floor: f32) -> String {
    if val <= floor {
        "-inf".to_string()
    } else {
        format!("{:+.1}", val)
    }
}

fn db_to_meter_t(db: f32) -> f32 {
    ((db + 60.0) / 60.0).clamp(0.0, 1.0)
}

fn draw_level_meter(ui: &mut egui::Ui, label: &str, peak_db: f32, rms_db: f32, hold_db: f32, width: f32) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(label)
                .monospace()
                .strong()
                .size(11.0)
                .color(Color32::from_rgb(165, 195, 215)),
        );
        let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 11.0), egui::Sense::hover());
        let p = ui.painter();

        p.rect_filled(rect, CornerRadius::same(2), Color32::from_rgb(18, 25, 36));

        let peak_t = db_to_meter_t(peak_db);
        let rms_t = db_to_meter_t(rms_db);
        let hold_t = db_to_meter_t(hold_db);

        // Draw segmented studio gradient bar up to peak
        let segments = 32;
        let seg_w = (rect.width() - (segments as f32 - 1.0) * 1.0) / (segments as f32);
        for i in 0..segments {
            let t0 = (i as f32) / (segments as f32);
            let t1 = ((i + 1) as f32) / (segments as f32);
            if t0 >= peak_t {
                break;
            }
            let col = if t1 > 0.90 {
                Color32::from_rgb(255, 68, 78) // > -6 dB red
            } else if t1 > 0.75 {
                Color32::from_rgb(255, 195, 58) // -15..-6 dB amber
            } else {
                Color32::from_rgb(38, 228, 175) // nominal cyan-green
            };
            let dimmed = if t0 > rms_t {
                Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 155)
            } else {
                col
            };
            let x = rect.left() + (i as f32) * (seg_w + 1.0);
            let srect = Rect::from_min_size(egui::pos2(x, rect.top() + 1.0), Vec2::new(seg_w, rect.height() - 2.0));
            p.rect_filled(srect, CornerRadius::same(1), dimmed);
        }

        // Peak-hold marker tick
        if hold_t > 0.01 {
            let hx = rect.left() + hold_t * rect.width();
            p.line_segment(
                [egui::pos2(hx, rect.top()), egui::pos2(hx, rect.bottom())],
                Stroke::new(2.0, Color32::from_rgb(235, 250, 255)),
            );
        }

        ui.label(
            RichText::new(format!("{:>5}", format_db(peak_db, -85.0)))
                .monospace()
                .size(10.5)
                .color(Color32::from_rgb(175, 200, 220)),
        );
    });
}

fn draw_correlation_bar(ui: &mut egui::Ui, corr: f32, width: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 12.0), egui::Sense::hover());
    let p = ui.painter();

    p.rect_filled(rect, CornerRadius::same(3), Color32::from_rgb(18, 25, 36));
    let mid_x = rect.center().x;

    // Center 0 tick
    p.line_segment(
        [egui::pos2(mid_x, rect.top()), egui::pos2(mid_x, rect.bottom())],
        Stroke::new(1.0, Color32::from_rgb(90, 115, 140)),
    );

    let t = corr.clamp(-1.0, 1.0);
    let fill_col = if t < 0.0 {
        Color32::from_rgb(255, 72, 78)
    } else if t < 0.35 {
        Color32::from_rgb(255, 195, 58)
    } else {
        Color32::from_rgb(45, 238, 210)
    };

    let bar_x0 = if t >= 0.0 { mid_x } else { mid_x + t * (rect.width() * 0.5) };
    let bar_x1 = if t >= 0.0 { mid_x + t * (rect.width() * 0.5) } else { mid_x };
    let bar_rect = Rect::from_min_max(
        egui::pos2(bar_x0, rect.top() + 2.0),
        egui::pos2(bar_x1, rect.bottom() - 2.0),
    );
    p.rect_filled(bar_rect, CornerRadius::same(2), fill_col);
    p.rect_stroke(
        rect,
        CornerRadius::same(3),
        Stroke::new(1.0, Color32::from_rgb(38, 56, 76)),
        egui::StrokeKind::Inside,
    );
}

fn draw_readout_card(ui: &mut egui::Ui, title: &str, value: String, color: Color32) {
    egui::Frame::NONE
        .fill(Color32::from_rgb(16, 22, 33))
        .corner_radius(CornerRadius::same(4))
        .stroke(Stroke::new(1.0, Color32::from_rgb(30, 46, 65)))
        .inner_margin(Margin::symmetric(10, 5))
        .show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(
                    RichText::new(title)
                        .size(9.5)
                        .strong()
                        .color(Color32::from_rgb(125, 152, 178)),
                );
                ui.label(RichText::new(value).monospace().strong().size(14.0).color(color));
            });
        });
}

fn draw_canvas_annotations(
    ctx: &egui::Context,
    screen: Rect,
    top_h: f32,
    bot_h: f32,
    settings: &ScopeSettings,
    sample_rate: u32,
) {
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::unique("canvas_annotations"),
    ));
    let usable_top = screen.top() + top_h;
    let usable_bot = screen.bottom() - bot_h;
    let usable_h = (usable_bot - usable_top).max(1.0);
    let label_col = Color32::from_rgba_unmultiplied(150, 215, 232, 165);
    let muted_col = Color32::from_rgba_unmultiplied(125, 175, 195, 130);
    let font_sm = egui::FontId::monospace(10.5);
    let font_xs = egui::FontId::monospace(9.5);

    // 1. CRT Phosphor Pane Annotations
    if settings.layout != LayoutMode::WaterfallOnly {
        let pane_w = if settings.layout == LayoutMode::SplitStudio {
            screen.width() * settings.split_ratio
        } else {
            screen.width()
        };
        let pane_rect = Rect::from_min_size(
            egui::pos2(screen.left(), usable_top),
            Vec2::new(pane_w, usable_h),
        );
        let tag = if settings.rotate_ms {
            "CRT PHOSPHOR · M/S 45°"
        } else {
            "CRT PHOSPHOR · XY L/R"
        };
        painter.text(
            egui::pos2(pane_rect.left() + 12.0, pane_rect.top() + 10.0),
            egui::Align2::LEFT_TOP,
            tag,
            font_sm.clone(),
            label_col,
        );

        let side = pane_w.min(usable_h) * 0.84;
        let c = pane_rect.center();
        let half = side * 0.44;
        if settings.rotate_ms {
            painter.text(egui::pos2(c.x, c.y - half), egui::Align2::CENTER_TOP, "+M", font_xs.clone(), muted_col);
            painter.text(egui::pos2(c.x - half, c.y), egui::Align2::LEFT_CENTER, "-S", font_xs.clone(), muted_col);
            painter.text(egui::pos2(c.x + half, c.y), egui::Align2::RIGHT_CENTER, "+S", font_xs.clone(), muted_col);
            painter.text(
                egui::pos2(c.x - half * 0.68, c.y - half * 0.68),
                egui::Align2::CENTER_CENTER,
                "L",
                font_xs.clone(),
                label_col,
            );
            painter.text(
                egui::pos2(c.x + half * 0.68, c.y - half * 0.68),
                egui::Align2::CENTER_CENTER,
                "R",
                font_xs.clone(),
                label_col,
            );
        } else {
            painter.text(egui::pos2(c.x + half, c.y - 6.0), egui::Align2::RIGHT_BOTTOM, "+L", font_xs.clone(), muted_col);
            painter.text(egui::pos2(c.x + 6.0, c.y - half), egui::Align2::LEFT_TOP, "+R", font_xs.clone(), muted_col);
        }
    }

    // 2. 3D Waterfall Pane Annotations & Log Frequency Axis Strip
    if settings.layout != LayoutMode::PhosphorOnly {
        let left_x = if settings.layout == LayoutMode::SplitStudio {
            screen.left() + screen.width() * settings.split_ratio
        } else {
            screen.left()
        };
        let wf_w = (screen.right() - left_x).max(1.0);
        painter.text(
            egui::pos2(left_x + 12.0, usable_top + 10.0),
            egui::Align2::LEFT_TOP,
            format!(
                "8192-PT GPU FFT · 3D WATERFALL · TILT {:+.1} dB/oct",
                settings.tilt_db_per_oct
            ),
            font_sm,
            label_col,
        );

        let f_min = 20.0f32;
        let f_max = (sample_rate as f32 * 0.48).min(20_000.0);
        let log_span = (f_max / f_min).ln();
        let freqs: &[(f32, &str)] = &[
            (30.0, "30Hz"),
            (60.0, "60Hz"),
            (100.0, "100Hz"),
            (250.0, "250Hz"),
            (500.0, "500Hz"),
            (1_000.0, "1kHz"),
            (2_500.0, "2.5kHz"),
            (5_000.0, "5kHz"),
            (10_000.0, "10kHz"),
            (16_000.0, "16kHz"),
        ];
        let pad_x = wf_w * 0.08;
        let axis_w = (wf_w - pad_x * 2.0).max(1.0);
        let y = usable_bot - 8.0;
        for &(hz, text) in freqs {
            if hz > f_max {
                continue;
            }
            let u = (hz / f_min).ln() / log_span;
            let x = left_x + pad_x + u * axis_w;
            painter.text(
                egui::pos2(x, y),
                egui::Align2::CENTER_BOTTOM,
                text,
                font_xs.clone(),
                muted_col,
            );
        }
    }
}

/// Generates a crisp 64x64 studio app icon (glowing cyan CRT ring + waveform) embedded in the binary.
fn build_window_icon() -> Option<Icon> {
    const S: usize = 64;
    let mut rgba = vec![0u8; S * S * 4];
    for y in 0..S {
        for x in 0..S {
            let nx = (x as f32 + 0.5) / (S as f32) * 2.0 - 1.0;
            let ny = (y as f32 + 0.5) / (S as f32) * 2.0 - 1.0;
            let r = (nx * nx + ny * ny).sqrt();

            // Rounded studio badge silhouette
            let qx = nx.abs() - 0.72;
            let qy = ny.abs() - 0.72;
            let sdf = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt() + qx.max(qy).min(0.0) - 0.22;
            let alpha = (1.0 - ((sdf + 0.02) / 0.04).clamp(0.0, 1.0)) * 255.0;

            let mut cr = 0.04f32;
            let mut cg = 0.07f32;
            let mut cb = 0.12f32;

            // Outer glowing CRT circle
            let ring = (-((r - 0.68) / 0.055).powi(2)).exp();
            cr += 0.08 * ring;
            cg += 0.92 * ring;
            cb += 0.88 * ring;

            // Lissajous figure-8 wave inside the CRT ring
            let wave_y = 0.38 * (nx * 4.5).sin() * (-(nx * 1.15).powi(2)).exp();
            let wave_y2 = -0.30 * (nx * 6.5 + 0.8).sin() * (-(nx * 1.15).powi(2)).exp();
            let d1 = (ny - wave_y).abs();
            let d2 = (ny - wave_y2).abs();
            let g1 = (-(d1 / 0.075).powi(2)).exp();
            let g2 = (-(d2 / 0.085).powi(2)).exp();
            cr += 0.25 * g1 + 0.85 * g2;
            cg += 1.00 * g1 + 0.35 * g2;
            cb += 0.92 * g1 + 0.95 * g2;

            let idx = (y * S + x) * 4;
            rgba[idx] = (cr.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 1] = (cg.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 2] = (cb.clamp(0.0, 1.0) * 255.0) as u8;
            rgba[idx + 3] = alpha.clamp(0.0, 255.0) as u8;
        }
    }
    Icon::from_rgba(rgba, S as u32, S as u32).ok()
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let event_loop = EventLoop::new().expect("failed to create winit event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop error");
}
