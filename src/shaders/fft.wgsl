const N: u32 = 8192u;
const LOG2_N: u32 = 13u;
const NUM_BINS: u32 = 512u;
const PI: f32 = 3.14159265358979323846;

struct FftParams {
    sample_rate: f32,
    tilt_db_per_oct: f32,
    smoothing: f32,
    write_row: u32,
    min_db: f32,
    max_db: f32,
    stage: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: FftParams;
@group(0) @binding(1) var<storage, read> input_samples: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> src_fft: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> dst_fft: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> prev_spectrum: array<f32>;
@group(0) @binding(5) var waterfall_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(64)
fn window_bitrev(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= N) {
        return;
    }
    let s_lr = input_samples[i];
    let mono = 0.5 * (s_lr.x + s_lr.y);
    let hann = 0.5 * (1.0 - cos(2.0 * PI * f32(i) / f32(N - 1u)));
    let rev = reverseBits(i) >> (32u - LOG2_N);
    dst_fft[rev] = vec2<f32>(mono * hann, 0.0);
}

@compute @workgroup_size(64)
fn fft_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= (N >> 1u)) {
        return;
    }
    let stage = params.stage;
    let half_len = 1u << stage;
    let group = tid >> stage;
    let k = tid & (half_len - 1u);
    let idx0 = (group << (stage + 1u)) + k;
    let idx1 = idx0 + half_len;

    let a = src_fft[idx0];
    let b = src_fft[idx1];

    let angle = -PI * f32(k) / f32(half_len);
    let tw = vec2<f32>(cos(angle), sin(angle));
    let t = vec2<f32>(b.x * tw.x - b.y * tw.y, b.x * tw.y + b.y * tw.x);

    dst_fft[idx0] = a + t;
    dst_fft[idx1] = a - t;
}

@compute @workgroup_size(64)
fn spectrum_log_row(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bin = gid.x;
    if (bin >= NUM_BINS) {
        return;
    }

    let f_min = 20.0;
    let f_max = min(20000.0, params.sample_rate * 0.48);
    let ratio = f_max / f_min;

    let t0 = f32(bin) / f32(NUM_BINS);
    let t1 = f32(bin + 1u) / f32(NUM_BINS);
    let tc = (f32(bin) + 0.5) / f32(NUM_BINS);

    let freq0 = f_min * pow(ratio, t0);
    let freq1 = f_min * pow(ratio, t1);
    let freq_c = f_min * pow(ratio, tc);

    let bin_scale = f32(N) / params.sample_rate;
    let k0 = freq0 * bin_scale;
    let k1 = freq1 * bin_scale;
    let kc = freq_c * bin_scale;

    let norm_factor = 1.0 / (f32(N) * 0.25); // 0 dBFS sine -> 1.0 after Hann window
    var mag: f32 = 0.0;

    if (k1 - k0 < 1.0) {
        let i0 = clamp(u32(floor(kc)), 1u, (N >> 1u) - 2u);
        let frac = fract(kc);
        let m0 = length(src_fft[i0]) * norm_factor;
        let m1 = length(src_fft[i0 + 1u]) * norm_factor;
        mag = mix(m0, m1, frac);
    } else {
        let start_k = clamp(u32(floor(k0)), 1u, (N >> 1u) - 1u);
        let end_k = clamp(u32(ceil(k1)), start_k, (N >> 1u) - 1u);
        var peak: f32 = 0.0;
        var sum_sq: f32 = 0.0;
        var count: f32 = 0.0;
        for (var k = start_k; k <= end_k; k = k + 1u) {
            let m = length(src_fft[k]) * norm_factor;
            peak = max(peak, m);
            sum_sq = sum_sq + m * m;
            count = count + 1.0;
        }
        let rms = sqrt(sum_sq / max(count, 1.0));
        // Blend RMS and peak for accurate mastering balance without losing narrow peaks
        mag = mix(rms, peak, 0.72);
    }

    var db = 20.0 * log(max(mag, 1e-6)) / log(10.0);
    // Tilt around 1 kHz pivot (e.g. +3.0 or +4.5 dB/oct makes pink noise flat)
    db = db + params.tilt_db_per_oct * log2(freq_c / 1000.0);

    let norm = clamp((db - params.min_db) / max(params.max_db - params.min_db, 1.0), 0.0, 1.0);
    let prev = prev_spectrum[bin];
    let smoothed = select(
        mix(norm, prev, params.smoothing),
        mix(prev, norm, 0.78),
        norm > prev
    );
    prev_spectrum[bin] = smoothed;

    textureStore(waterfall_tex, vec2<i32>(i32(bin), i32(params.write_row)), vec4<f32>(smoothed, 0.0, 0.0, 1.0));
}
