struct PhosphorParams {
    decay: f32,
    beam_intensity: f32,
    beam_radius: f32,
    rotate_ms: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    gain: f32,
    sample_offset: u32,
    segment_count: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<uniform> params: PhosphorParams;
@group(0) @binding(1) var<storage, read> samples: array<vec2<f32>>;

// ---- Pass A: Exponential Phosphor Decay ----
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0)
    );
    return vec4<f32>(pos[vid], 0.0, 1.0);
}

@fragment
fn fs_decay() -> @location(0) vec4<f32> {
    // Subtle differential RGB decay mimics real P31/P7 CRT phosphor afterglow
    let d = clamp(params.decay, 0.0, 0.995);
    return vec4<f32>(d * 0.975, d, d * 0.99, 1.0);
}

// ---- Pass B: Electron-Beam Instanced Capsules ----
struct BeamOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) local_pos: vec2<f32>,
    @location(1) half_len: f32,
    @location(2) energy: f32,
};

fn map_xy(s: vec2<f32>) -> vec2<f32> {
    let scaled = s * params.gain;
    if (params.rotate_ms > 0.5) {
        // Pro Mastering 45 deg M/S vectorscope:
        // Mono (L == R) -> vertical center, L -> upper-left, R -> upper-right
        let x = (scaled.y - scaled.x) * 0.70710678;
        let y = (scaled.x + scaled.y) * 0.70710678;
        return vec2<f32>(x, y) * 0.78;
    } else {
        // Classic Lissajous XY (X = Left, Y = Right)
        return scaled * 0.85;
    }
}

@vertex
fn vs_beam(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) iid: u32,
) -> BeamOut {
    let idx0 = params.sample_offset + iid;
    let idx1 = idx0 + 1u;

    let p0 = map_xy(samples[idx0]);
    let p1 = map_xy(samples[idx1]);

    let delta = p1 - p0;
    let seg_len = length(delta);
    let dir = select(vec2<f32>(1.0, 0.0), delta / seg_len, seg_len > 1e-5);
    let perp = vec2<f32>(-dir.y, dir.x);

    let radius = params.beam_radius;
    let half_len = seg_len * 0.5;
    let center = (p0 + p1) * 0.5;

    // Quad corner coordinates in local capsule space
    var corner = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
    );
    let c = corner[vid];
    let along = c.x * (half_len + radius);
    let across = c.y * radius;
    let world = center + dir * along + perp * across;

    // Velocity-dependent CRT beam intensity + subtle recency taper
    let age_t = f32(iid) / max(f32(params.segment_count), 1.0);
    let recency = mix(0.55, 1.15, age_t);
    let velocity_energy = (1.0 / (seg_len * 34.0 + 0.28)) * params.beam_intensity * recency;

    var out: BeamOut;
    out.clip_pos = vec4<f32>(world, 0.0, 1.0);
    out.local_pos = vec2<f32>(along, across);
    out.half_len = half_len;
    out.energy = velocity_energy;
    return out;
}

@fragment
fn fs_beam(in: BeamOut) -> @location(0) vec4<f32> {
    let dx = max(0.0, abs(in.local_pos.x) - in.half_len);
    let dy = in.local_pos.y;
    let dist = length(vec2<f32>(dx, dy));

    let core_sigma = max(params.beam_radius * 0.24, 1e-4);
    let halo_sigma = max(params.beam_radius * 0.62, 1e-4);

    let core = exp(-pow(dist / core_sigma, 2.0));
    let halo = 0.28 * exp(-pow(dist / halo_sigma, 2.0));

    let base_col = vec3<f32>(params.color_r, params.color_g, params.color_b);
    // Hot white-core electron burn + colored phosphor halo
    let col = base_col * (core * 1.1 + halo) + vec3<f32>(0.45, 0.55, 0.55) * (core * core * 0.85);
    return vec4<f32>(col * in.energy, 1.0);
}
