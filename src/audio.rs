use pipewire as pw;
use pw::{properties::properties, spa};
use rtrb::{Consumer, Producer, RingBuffer};
use spa::param::format::{MediaSubtype, MediaType};
use spa::param::format_utils;
use spa::pod::Pod;
use std::cell::RefCell;
use std::f32::consts::TAU;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioNodeInfo {
    pub id: u32,
    pub name: String,
    pub label: String,
    pub media_class: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestSignalMode {
    LissajousHarmonic,
    TranceKickSynth,
    FreqSweep,
}

impl TestSignalMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::LissajousHarmonic => "Test: Lissajous Harmonics",
            Self::TranceKickSynth => "Test: Mastering Kick + Stereo Synth",
            Self::FreqSweep => "Test: 20Hz–20kHz Log Sweep",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AudioSource {
    DefaultMonitor,
    PipeWireNode { name: String, label: String },
    TestSignal(TestSignalMode),
}

impl AudioSource {
    pub fn display_name(&self) -> String {
        match self {
            Self::DefaultMonitor => "PipeWire: System Default Output (Monitor)".to_string(),
            Self::PipeWireNode { label, .. } => format!("PipeWire: {}", label),
            Self::TestSignal(m) => m.label().to_string(),
        }
    }
}

enum PwCommand {
    Reconnect {
        target: Option<String>,
        producer: Producer<[f32; 2]>,
    },
    Quit,
}

struct StreamHandle {
    _stream: pw::stream::StreamRc,
    _listener: pw::stream::StreamListener<StreamUserData>,
}

struct StreamUserData {
    format: spa::param::audio::AudioInfoRaw,
    producer: Producer<[f32; 2]>,
    sample_rate: Arc<AtomicU32>,
}

pub struct AudioCapture {
    consumer: Consumer<[f32; 2]>,
    nodes: Arc<Mutex<Vec<AudioNodeInfo>>>,
    sample_rate: Arc<AtomicU32>,
    pw_connected: Arc<AtomicBool>,
    pw_tx: Option<pw::channel::Sender<PwCommand>>,
    pub source: AudioSource,
    synth_phase: f64,
}

impl AudioCapture {
    pub fn new() -> Self {
        let (producer, consumer) = RingBuffer::<[f32; 2]>::new(65_536);
        let nodes = Arc::new(Mutex::new(Vec::<AudioNodeInfo>::new()));
        let sample_rate = Arc::new(AtomicU32::new(48_000));
        let pw_connected = Arc::new(AtomicBool::new(false));

        let (pw_tx, pw_rx) = pw::channel::channel::<PwCommand>();

        let nodes_bg = Arc::clone(&nodes);
        let sr_bg = Arc::clone(&sample_rate);
        let conn_bg = Arc::clone(&pw_connected);

        thread::Builder::new()
            .name("pipewire-capture".to_string())
            .spawn(move || {
                if let Err(err) = run_pipewire_loop(producer, pw_rx, nodes_bg, sr_bg, conn_bg) {
                    log::error!("PipeWire capture thread error: {err}");
                }
            })
            .expect("failed to spawn PipeWire capture thread");

        Self {
            consumer,
            nodes,
            sample_rate,
            pw_connected,
            pw_tx: Some(pw_tx),
            source: AudioSource::DefaultMonitor,
            synth_phase: 0.0,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        match self.source {
            AudioSource::TestSignal(_) => 48_000,
            _ => self.sample_rate.load(Ordering::Relaxed).max(8_000),
        }
    }

    pub fn is_pipewire_connected(&self) -> bool {
        self.pw_connected.load(Ordering::Relaxed)
    }

    pub fn available_nodes(&self) -> Vec<AudioNodeInfo> {
        self.nodes
            .lock()
            .map(|guard| guard.clone())
            .unwrap_or_default()
    }

    pub fn set_source(&mut self, next: AudioSource) {
        if self.source == next {
            return;
        }
        self.source = next.clone();
        match next {
            AudioSource::DefaultMonitor => {
                let (producer, consumer) = RingBuffer::<[f32; 2]>::new(65_536);
                self.consumer = consumer;
                if let Some(tx) = &self.pw_tx {
                    let _ = tx.send(PwCommand::Reconnect {
                        target: None,
                        producer,
                    });
                }
            }
            AudioSource::PipeWireNode { name, .. } => {
                let (producer, consumer) = RingBuffer::<[f32; 2]>::new(65_536);
                self.consumer = consumer;
                if let Some(tx) = &self.pw_tx {
                    let _ = tx.send(PwCommand::Reconnect {
                        target: Some(name),
                        producer,
                    });
                }
            }
            AudioSource::TestSignal(_) => {}
        }
    }

    /// Drains all newly arrived stereo frames into `out`.
    pub fn drain_samples(&mut self, dt_sec: f32, out: &mut Vec<[f32; 2]>) {
        out.clear();

        // Always drain the PipeWire ring buffer so it never stalls when switching back
        let mut pw_samples = Vec::new();
        let avail = self.consumer.slots();
        if avail > 0 {
            if let Ok(chunk) = self.consumer.read_chunk(avail) {
                let (s1, s2) = chunk.as_slices();
                pw_samples.extend_from_slice(s1);
                pw_samples.extend_from_slice(s2);
                chunk.commit_all();
            }
        }

        match self.source {
            AudioSource::TestSignal(mode) => {
                let sr = 48_000.0f64;
                let count = ((dt_sec.clamp(0.001, 0.05) as f64) * sr).round() as usize;
                out.reserve(count);
                for _ in 0..count {
                    let t = self.synth_phase;
                    self.synth_phase += 1.0 / sr;
                    out.push(synth_sample(mode, t));
                }
            }
            _ => {
                out.extend_from_slice(&pw_samples);
            }
        }
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        if let Some(tx) = self.pw_tx.take() {
            let _ = tx.send(PwCommand::Quit);
        }
    }
}

fn synth_sample(mode: TestSignalMode, t: f64) -> [f32; 2] {
    let tf = t as f32;
    match mode {
        TestSignalMode::LissajousHarmonic => {
            // Evolving rich harmonic Lissajous figure + gentle stereo chorus + sub
            let f0 = 110.0;
            let ratio_r = 1.5 + 0.004 * (tf * 0.45).sin();
            let ph_drift = (tf * 0.35).sin() * 1.4;
            let l = 0.48 * (TAU * f0 * tf).sin()
                + 0.24 * (TAU * f0 * 3.0 * tf + 0.4).sin()
                + 0.14 * (TAU * 55.0 * tf).sin()
                + 0.08 * (TAU * 1760.0 * tf).sin() * (0.5 + 0.5 * (tf * 2.0).sin());
            let r = 0.48 * (TAU * f0 * ratio_r * tf + ph_drift).sin()
                + 0.24 * (TAU * f0 * 2.0 * tf - 0.5).sin()
                + 0.14 * (TAU * 55.0 * tf).sin()
                + 0.08 * (TAU * 2640.0 * tf).sin() * (0.5 + 0.5 * (tf * 2.0 + 1.1).sin());
            [l * 0.78, r * 0.78]
        }
        TestSignalMode::TranceKickSynth => {
            // 136 BPM punchy kick + supersaw stereo chord + crisp offbeat hi-hat
            let bpm = 136.0;
            let beat = tf * (bpm / 60.0);
            let frac = beat.fract();

            // Sub kick: exponential pitch drop 160Hz -> 44Hz
            let kick_env = (-frac * 11.5).exp();
            let kick_phase = TAU * (44.0 * frac + (116.0 / 28.0) * (1.0 - (-frac * 28.0).exp()));
            let kick = kick_phase.sin() * kick_env * 0.62;

            // Stereo pumping supersaw pad (sidechained to kick)
            let sidechain = 1.0 - (-frac * 7.5).exp();
            let chord_idx = ((beat / 4.0).floor() as i32).rem_euclid(4);
            let root = match chord_idx {
                0 => 146.83, // D3
                1 => 116.54, // Bb2
                2 => 130.81, // C3
                _ => 110.00, // A2
            };
            let freqs = [root, root * 1.1892, root * 1.4983, root * 2.0, root * 3.0];
            let mut pad_l = 0.0f32;
            let mut pad_r = 0.0f32;
            for (i, &f) in freqs.iter().enumerate() {
                let detune = 1.0 + 0.0055 * (i as f32 - 2.0);
                let pl = ((tf * f * detune).fract() * 2.0 - 1.0) * 0.11;
                let pr = ((tf * f / detune + 0.17).fract() * 2.0 - 1.0) * 0.11;
                pad_l += pl;
                pad_r += pr;
            }
            // Melodic shimmer arp
            let step = ((beat * 4.0).floor() as usize) % 4;
            let arp_f = freqs[step] * 4.0;
            let arp_env = (-(beat * 4.0).fract() * 9.0).exp();
            let arp = (TAU * arp_f * tf).sin() * arp_env * 0.16;

            // Offbeat metallic hi-hat shimmer around 8-12kHz
            let hat_frac = (beat + 0.5).fract();
            let hat_env = (-hat_frac * 36.0).exp();
            let hat_l = ((TAU * 8340.0 * tf).sin() + (TAU * 11720.0 * tf).sin()) * hat_env * 0.08;
            let hat_r = ((TAU * 9120.0 * tf).sin() + (TAU * 12890.0 * tf).sin()) * hat_env * 0.08;

            let l = (kick + pad_l * sidechain + arp * 0.8 + hat_l).clamp(-0.98, 0.98);
            let r = (kick + pad_r * sidechain + arp * 1.2 + hat_r).clamp(-0.98, 0.98);
            [l, r]
        }
        TestSignalMode::FreqSweep => {
            // 6-second logarithmic sweep 25Hz -> 18kHz with 90-degree quadrature stereo phase
            let period = 6.0f64;
            let u = (t % period) / period;
            let f0 = 25.0f64;
            let f1 = 18_000.0f64;
            let k = (f1 / f0).ln();
            let inst_phase = 2.0 * std::f64::consts::PI * f0 * period / k * ((k * u).exp() - 1.0);
            let l = (inst_phase.sin() * 0.65) as f32;
            let r = ((inst_phase + 0.65 * std::f64::consts::FRAC_PI_2).sin() * 0.65) as f32;
            [l, r]
        }
    }
}

fn create_capture_stream(
    core: &pw::core::CoreRc,
    target_object: Option<&str>,
    producer: Producer<[f32; 2]>,
    sample_rate: Arc<AtomicU32>,
    active: Arc<AtomicBool>,
) -> Result<StreamHandle, pw::Error> {
    let mut props = properties! {
        *pw::keys::MEDIA_TYPE => "Audio",
        *pw::keys::MEDIA_CATEGORY => "Capture",
        *pw::keys::MEDIA_ROLE => "Music",
        *pw::keys::NODE_NAME => "vulkanscope-pro",
        *pw::keys::APP_NAME => "VulkanScope Pro",
    };

    if let Some(target) = target_object {
        props.insert(*pw::keys::TARGET_OBJECT, target);
        props.insert(*pw::keys::STREAM_CAPTURE_SINK, "true");
    } else {
        props.insert(*pw::keys::STREAM_CAPTURE_SINK, "true");
    }

    let stream = pw::stream::StreamRc::new(core.clone(), "vulkanscope-capture", props)?;

    let user_data = StreamUserData {
        format: Default::default(),
        producer,
        sample_rate,
    };

    let active_state = Arc::clone(&active);
    let listener = stream
        .add_local_listener_with_user_data(user_data)
        .state_changed(move |_, _, _, new_state| {
            let is_streaming = matches!(new_state, pw::stream::StreamState::Streaming);
            active_state.store(is_streaming, Ordering::Relaxed);
        })
        .param_changed(|_, data, id, param| {
            let Some(param) = param else {
                return;
            };
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let Ok((media_type, media_subtype)) = format_utils::parse_format(param) else {
                return;
            };
            if media_type != MediaType::Audio || media_subtype != MediaSubtype::Raw {
                return;
            }
            if data.format.parse(param).is_ok() {
                let rate = data.format.rate();
                if rate > 0 {
                    data.sample_rate.store(rate, Ordering::Relaxed);
                }
            }
        })
        .process(|stream, data| {
            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };
            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }

            let d = &mut datas[0];
            let chunk_size = d.chunk().size() as usize;
            let chunk_offset = d.chunk().offset() as usize;
            let n_channels = data.format.channels().max(1) as usize;

            let Some(raw) = d.data() else {
                return;
            };
            if chunk_offset >= raw.len() || chunk_size < 4 {
                return;
            }
            let end = (chunk_offset + chunk_size).min(raw.len());
            let bytes = &raw[chunk_offset..end];

            let total_floats = bytes.len() / 4;
            let n_frames = total_floats / n_channels;
            if n_frames == 0 {
                return;
            }

            // Lock-free write into rtrb ring buffer (zero allocations, zero mutexes!)
            let avail = data.producer.slots().min(n_frames);
            if avail == 0 {
                return;
            }
            if let Ok(mut wchunk) = data.producer.write_chunk_uninit(avail) {
                let (s1, s2) = wchunk.as_mut_slices();
                let mut frame_idx = 0usize;

                for slot in s1.iter_mut() {
                    let base = frame_idx * n_channels * 4;
                    let l = f32::from_le_bytes([
                        bytes[base],
                        bytes[base + 1],
                        bytes[base + 2],
                        bytes[base + 3],
                    ]);
                    let r = if n_channels >= 2 {
                        let rb = base + 4;
                        f32::from_le_bytes([bytes[rb], bytes[rb + 1], bytes[rb + 2], bytes[rb + 3]])
                    } else {
                        l
                    };
                    slot.write([l, r]);
                    frame_idx += 1;
                }
                for slot in s2.iter_mut() {
                    let base = frame_idx * n_channels * 4;
                    let l = f32::from_le_bytes([
                        bytes[base],
                        bytes[base + 1],
                        bytes[base + 2],
                        bytes[base + 3],
                    ]);
                    let r = if n_channels >= 2 {
                        let rb = base + 4;
                        f32::from_le_bytes([bytes[rb], bytes[rb + 1], bytes[rb + 2], bytes[rb + 3]])
                    } else {
                        l
                    };
                    slot.write([l, r]);
                    frame_idx += 1;
                }
                unsafe {
                    wchunk.commit_all();
                }
            }
        })
        .register()?;

    let mut audio_info = spa::param::audio::AudioInfoRaw::new();
    audio_info.set_format(spa::param::audio::AudioFormat::F32LE);
    let obj = spa::pod::Object {
        type_: spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
        id: spa::param::ParamType::EnumFormat.as_raw(),
        properties: audio_info.into(),
    };
    let values: Vec<u8> = spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(obj),
    )
    .expect("failed to serialize SPA audio format pod")
    .0
    .into_inner();

    let mut params = [Pod::from_bytes(&values).expect("valid SPA pod")];

    stream.connect(
        spa::utils::Direction::Input,
        None,
        pw::stream::StreamFlags::AUTOCONNECT
            | pw::stream::StreamFlags::MAP_BUFFERS
            | pw::stream::StreamFlags::RT_PROCESS,
        &mut params,
    )?;

    Ok(StreamHandle {
        _stream: stream,
        _listener: listener,
    })
}

fn run_pipewire_loop(
    producer: Producer<[f32; 2]>,
    pw_rx: pw::channel::Receiver<PwCommand>,
    nodes: Arc<Mutex<Vec<AudioNodeInfo>>>,
    sample_rate: Arc<AtomicU32>,
    active: Arc<AtomicBool>,
) -> Result<(), pw::Error> {
    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None)?;
    let context = pw::context::ContextRc::new(&mainloop, None)?;
    let core = context.connect_rc(None)?;
    let registry = core.get_registry_rc()?;

    // Track available PipeWire sinks & playback streams for the UI node selector
    let nodes_add = Arc::clone(&nodes);
    let nodes_rem = Arc::clone(&nodes);
    let _reg_listener = registry
        .add_listener_local()
        .global(move |global| {
            if global.type_ != pw::types::ObjectType::Node {
                return;
            }
            let Some(props) = global.props else {
                return;
            };
            let media_class = props.get("media.class").unwrap_or("");
            if media_class != "Audio/Sink" && media_class != "Stream/Output/Audio" {
                return;
            }
            let name = props.get("node.name").unwrap_or("").to_string();
            if name.is_empty() || name == "vulkanscope-pro" {
                return;
            }
            let desc = props
                .get("node.description")
                .or_else(|| props.get("application.name"))
                .unwrap_or(&name)
                .to_string();
            let prefix = if media_class == "Audio/Sink" {
                "[Sink] "
            } else {
                "[App] "
            };
            let info = AudioNodeInfo {
                id: global.id,
                name,
                label: format!("{prefix}{desc}"),
                media_class: media_class.to_string(),
            };
            if let Ok(mut list) = nodes_add.lock() {
                if !list.iter().any(|n| n.id == info.id) {
                    list.push(info);
                    list.sort_by(|a, b| a.label.cmp(&b.label));
                }
            }
        })
        .global_remove(move |id| {
            if let Ok(mut list) = nodes_rem.lock() {
                list.retain(|n| n.id != id);
            }
        })
        .register();

    let current_stream: Rc<RefCell<Option<StreamHandle>>> = Rc::new(RefCell::new(None));

    match create_capture_stream(
        &core,
        None,
        producer,
        Arc::clone(&sample_rate),
        Arc::clone(&active),
    ) {
        Ok(handle) => {
            *current_stream.borrow_mut() = Some(handle);
        }
        Err(err) => {
            log::warn!("Initial PipeWire stream connect failed: {err}");
        }
    }

    let mainloop_quit = mainloop.clone();
    let core_cmd = core.clone();
    let stream_slot = Rc::clone(&current_stream);
    let sr_cmd = Arc::clone(&sample_rate);
    let active_cmd = Arc::clone(&active);
    let _rx_attached = pw_rx.attach(mainloop.loop_(), move |cmd| match cmd {
        PwCommand::Quit => {
            *stream_slot.borrow_mut() = None;
            mainloop_quit.quit();
        }
        PwCommand::Reconnect { target, producer } => {
            *stream_slot.borrow_mut() = None;
            active_cmd.store(false, Ordering::Relaxed);
            match create_capture_stream(
                &core_cmd,
                target.as_deref(),
                producer,
                Arc::clone(&sr_cmd),
                Arc::clone(&active_cmd),
            ) {
                Ok(handle) => {
                    *stream_slot.borrow_mut() = Some(handle);
                }
                Err(err) => {
                    log::warn!("PipeWire reconnect failed: {err}");
                }
            }
        }
    });

    mainloop.run();
    *current_stream.borrow_mut() = None;
    Ok(())
}
