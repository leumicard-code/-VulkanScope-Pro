use ebur128::{EbuR128, Mode};

pub const FFT_SIZE: usize = 8192;

#[derive(Clone, Copy, Debug)]
pub struct MasteringMetrics {
    pub peak_l_db: f32,
    pub peak_r_db: f32,
    pub hold_l_db: f32,
    pub hold_r_db: f32,
    pub rms_l_db: f32,
    pub rms_r_db: f32,
    pub correlation: f32,      // -1.0 (out of phase) .. +1.0 (mono)
    pub stereo_width: f32,     // 0.0 (mono) .. 1.0 (pure side)
    pub lufs_momentary: f32,   // 400ms EBU R128
    pub lufs_short_term: f32,  // 3s EBU R128
    pub true_peak_db: f32,     // Max true peak dBTP
    pub clipped: bool,
    pub has_signal: bool,
}

impl Default for MasteringMetrics {
    fn default() -> Self {
        Self {
            peak_l_db: -90.0,
            peak_r_db: -90.0,
            hold_l_db: -90.0,
            hold_r_db: -90.0,
            rms_l_db: -90.0,
            rms_r_db: -90.0,
            correlation: 1.0,
            stereo_width: 0.0,
            lufs_momentary: -70.0,
            lufs_short_term: -70.0,
            true_peak_db: -90.0,
            clipped: false,
            has_signal: false,
        }
    }
}

pub struct DspProcessor {
    /// Rolling chronological stereo buffer of exactly FFT_SIZE frames.
    window: Vec<[f32; 2]>,
    interleaved_scratch: Vec<f32>,
    ebur128: Option<EbuR128>,
    ebur_rate: u32,
    pub metrics: MasteringMetrics,
    hold_timer_l: f32,
    hold_timer_r: f32,
    clip_timer: f32,
    /// Number of fresh samples pushed on the last tick (for accurate CRT beam segment count).
    pub last_fresh_count: usize,
}

impl DspProcessor {
    pub fn new(sample_rate: u32) -> Self {
        let ebur128 = EbuR128::new(2, sample_rate, Mode::M | Mode::S | Mode::TRUE_PEAK).ok();
        Self {
            window: vec![[0.0, 0.0]; FFT_SIZE],
            interleaved_scratch: Vec::with_capacity(8192),
            ebur128,
            ebur_rate: sample_rate,
            metrics: MasteringMetrics::default(),
            hold_timer_l: 0.0,
            hold_timer_r: 0.0,
            clip_timer: 0.0,
            last_fresh_count: 0,
        }
    }

    pub fn window(&self) -> &[[f32; 2]] {
        &self.window
    }

    pub fn reset_meters(&mut self) {
        self.metrics = MasteringMetrics::default();
        self.hold_timer_l = 0.0;
        self.hold_timer_r = 0.0;
        self.clip_timer = 0.0;
        if let Some(eb) = &mut self.ebur128 {
            eb.reset();
        }
    }

    pub fn push_samples(&mut self, incoming: &[[f32; 2]], sample_rate: u32, dt_sec: f32) {
        if sample_rate != self.ebur_rate && sample_rate >= 8_000 {
            self.ebur128 = EbuR128::new(2, sample_rate, Mode::M | Mode::S | Mode::TRUE_PEAK).ok();
            self.ebur_rate = sample_rate;
        }

        let n = incoming.len();
        self.last_fresh_count = n;

        if n > 0 {
            if n >= FFT_SIZE {
                self.window.copy_from_slice(&incoming[n - FFT_SIZE..]);
            } else {
                self.window.copy_within(n.., 0);
                self.window[FFT_SIZE - n..].copy_from_slice(incoming);
            }

            // Feed EBU R128 loudness & true-peak engine
            if let Some(eb) = &mut self.ebur128 {
                self.interleaved_scratch.clear();
                self.interleaved_scratch.reserve(n * 2);
                for s in incoming {
                    self.interleaved_scratch.push(s[0]);
                    self.interleaved_scratch.push(s[1]);
                }
                let _ = eb.add_frames_f32(&self.interleaved_scratch);
            }
        } else {
            // Gentle fade-out on complete stream silence so scope & waterfall settle cleanly
            for s in &mut self.window {
                s[0] *= 0.88;
                s[1] *= 0.88;
            }
        }

        // Compute mastering metrics over recent slice (~2048 samples = ~42ms)
        let eval_len = 2048.min(self.window.len());
        let slice = &self.window[self.window.len() - eval_len..];

        let mut peak_l = 0.0f32;
        let mut peak_r = 0.0f32;
        let mut sum_l2 = 0.0f64;
        let mut sum_r2 = 0.0f64;
        let mut sum_lr = 0.0f64;
        let mut sum_mid2 = 0.0f64;
        let mut sum_side2 = 0.0f64;

        for &[l, r] in slice {
            let al = l.abs();
            let ar = r.abs();
            if al > peak_l {
                peak_l = al;
            }
            if ar > peak_r {
                peak_r = ar;
            }
            let lf = l as f64;
            let rf = r as f64;
            sum_l2 += lf * lf;
            sum_r2 += rf * rf;
            sum_lr += lf * rf;
            let m = 0.5 * (lf + rf);
            let s = 0.5 * (lf - rf);
            sum_mid2 += m * m;
            sum_side2 += s * s;
        }

        let inv_n = 1.0 / (eval_len as f64);
        let rms_l = (sum_l2 * inv_n).sqrt() as f32;
        let rms_r = (sum_r2 * inv_n).sqrt() as f32;

        let inst_peak_l_db = lin_to_db(peak_l);
        let inst_peak_r_db = lin_to_db(peak_r);
        let inst_rms_l_db = lin_to_db(rms_l);
        let inst_rms_r_db = lin_to_db(rms_r);

        // Smooth meter ballistics (fast attack, studio release ~24 dB/s)
        let decay_db = 28.0 * dt_sec;
        self.metrics.peak_l_db = if inst_peak_l_db > self.metrics.peak_l_db {
            inst_peak_l_db
        } else {
            (self.metrics.peak_l_db - decay_db).max(inst_peak_l_db)
        };
        self.metrics.peak_r_db = if inst_peak_r_db > self.metrics.peak_r_db {
            inst_peak_r_db
        } else {
            (self.metrics.peak_r_db - decay_db).max(inst_peak_r_db)
        };

        let rms_alpha = (dt_sec * 12.0).clamp(0.05, 0.8);
        self.metrics.rms_l_db += (inst_rms_l_db - self.metrics.rms_l_db) * rms_alpha;
        self.metrics.rms_r_db += (inst_rms_r_db - self.metrics.rms_r_db) * rms_alpha;

        // Peak-hold ticks (1.5s hold, then smooth fall)
        if inst_peak_l_db >= self.metrics.hold_l_db {
            self.metrics.hold_l_db = inst_peak_l_db;
            self.hold_timer_l = 1.5;
        } else if self.hold_timer_l > 0.0 {
            self.hold_timer_l -= dt_sec;
        } else {
            self.metrics.hold_l_db = (self.metrics.hold_l_db - 18.0 * dt_sec).max(inst_peak_l_db);
        }

        if inst_peak_r_db >= self.metrics.hold_r_db {
            self.metrics.hold_r_db = inst_peak_r_db;
            self.hold_timer_r = 1.5;
        } else if self.hold_timer_r > 0.0 {
            self.hold_timer_r -= dt_sec;
        } else {
            self.metrics.hold_r_db = (self.metrics.hold_r_db - 18.0 * dt_sec).max(inst_peak_r_db);
        }

        // Clip indicator
        if peak_l >= 0.998 || peak_r >= 0.998 {
            self.clip_timer = 2.0;
        } else if self.clip_timer > 0.0 {
            self.clip_timer -= dt_sec;
        }
        self.metrics.clipped = self.clip_timer > 0.0;
        self.metrics.has_signal = peak_l > 1e-4 || peak_r > 1e-4;

        // Phase correlation (-1..+1) & Stereo Width
        let denom = (sum_l2 * sum_r2).sqrt();
        let target_corr = if denom > 1e-10 {
            (sum_lr / denom).clamp(-1.0, 1.0) as f32
        } else {
            1.0
        };
        let corr_alpha = (dt_sec * 10.0).clamp(0.05, 0.5);
        self.metrics.correlation += (target_corr - self.metrics.correlation) * corr_alpha;

        let ms_total = sum_mid2 + sum_side2;
        let target_width = if ms_total > 1e-10 {
            (sum_side2 / ms_total).sqrt().clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
        self.metrics.stereo_width += (target_width - self.metrics.stereo_width) * corr_alpha;

        // Read EBU R128 LUFS & True Peak
        if let Some(eb) = &self.ebur128 {
            if let Ok(m) = eb.loudness_momentary() {
                self.metrics.lufs_momentary = if m.is_finite() {
                    (m as f32).clamp(-70.0, 6.0)
                } else {
                    -70.0
                };
            }
            if let Ok(s) = eb.loudness_shortterm() {
                self.metrics.lufs_short_term = if s.is_finite() {
                    (s as f32).clamp(-70.0, 6.0)
                } else {
                    -70.0
                };
            }
            let tp_l = eb.true_peak(0).unwrap_or(0.0) as f32;
            let tp_r = eb.true_peak(1).unwrap_or(0.0) as f32;
            self.metrics.true_peak_db = lin_to_db(tp_l.max(tp_r));
        }
    }
}

#[inline]
pub fn lin_to_db(v: f32) -> f32 {
    if v <= 1e-5 {
        -90.0
    } else {
        (20.0 * v.log10()).clamp(-90.0, 6.0)
    }
}
