const WF_WIDTH: u32 = 512u;
const WF_HEIGHT: u32 = 256u;

struct WaterfallParams {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    grid_dims: vec4<u32>,     // x = GX, y = GZ, z = write_row, w = palette
    visual: vec4<f32>,        // x = height_scale, y = glow_boost, z = wireframe_mix, w = flow_toward (1.0 = toward camera)
    freq_range: vec4<f32>,    // x = f_min, y = f_max, z = time_sec, w = unused
};

@group(0) @binding(0) var<uniform> params: WaterfallParams;
@group(0) @binding(1) var waterfall_tex: texture_2d<f32>;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,      // x = freq [0..1], y = age [0..1] (0 = newest)
    @location(3) amp: f32,
};

fn fetch_ring(u_norm: f32, age_norm: f32) -> f32 {
    let fx = clamp(u_norm, 0.0, 1.0) * f32(WF_WIDTH - 1u);
    let x0 = u32(floor(fx));
    let x1 = min(x0 + 1u, WF_WIDTH - 1u);
    let tx = fract(fx);

    let history_f = clamp(age_norm, 0.0, 1.0) * f32(WF_HEIGHT - 2u);
    let age0 = u32(floor(history_f));
    let age1 = min(age0 + 1u, WF_HEIGHT - 2u);
    let ty = fract(history_f);

    let write_row = params.grid_dims.z;
    let r0 = (write_row + WF_HEIGHT - (age0 % WF_HEIGHT)) % WF_HEIGHT;
    let r1 = (write_row + WF_HEIGHT - (age1 % WF_HEIGHT)) % WF_HEIGHT;

    let v00 = textureLoad(waterfall_tex, vec2<i32>(i32(x0), i32(r0)), 0).r;
    let v10 = textureLoad(waterfall_tex, vec2<i32>(i32(x1), i32(r0)), 0).r;
    let v01 = textureLoad(waterfall_tex, vec2<i32>(i32(x0), i32(r1)), 0).r;
    let v11 = textureLoad(waterfall_tex, vec2<i32>(i32(x1), i32(r1)), 0).r;

    return mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
}

fn terrain_height(u: f32, age: f32) -> f32 {
    let a = fetch_ring(u, age);
    // Sculptural response curve keeps valleys crisp and lets transients spike cleanly
    return pow(max(a, 0.0), 1.55) * params.visual.x;
}

@vertex
fn vs_waterfall(@builtin(vertex_index) vid: u32) -> VsOut {
    let gx = params.grid_dims.x;
    let gz = params.grid_dims.y;

    let ix = vid % gx;
    let iz = vid / gx;

    let u = f32(ix) / f32(gx - 1u);
    let v = f32(iz) / f32(gz - 1u);

    // Flow toward camera: newest spawns at horizon (z = -depth/2) and rolls forward to z = +depth/2
    // Flow receding: newest sits at front edge (z = +depth/2) and travels away to horizon
    let age = select(1.0 - v, v, params.visual.w > 0.5);

    let width = 8.4;
    let depth = 7.4;

    let x = (u - 0.5) * width;
    let z = (v - 0.5) * depth;

    let amp = fetch_ring(u, age);
    let y = pow(max(amp, 0.0), 1.55) * params.visual.x;

    // Finite-difference surface normal for smooth studio lighting
    let du = 1.0 / f32(gx - 1u);
    let dv = 1.0 / f32(gz - 1u);
    let age_p = select(1.0 - min(v + dv, 1.0), min(v + dv, 1.0), params.visual.w > 0.5);
    let age_m = select(1.0 - max(v - dv, 0.0), max(v - dv, 0.0), params.visual.w > 0.5);

    let hL = terrain_height(max(u - du, 0.0), age);
    let hR = terrain_height(min(u + du, 1.0), age);
    let hD = terrain_height(u, age_m);
    let hU = terrain_height(u, age_p);

    let dx_step = 2.0 * du * width;
    let dz_step = 2.0 * dv * depth;
    let tangent_x = vec3<f32>(dx_step, hR - hL, 0.0);
    let tangent_z = vec3<f32>(0.0, hU - hD, dz_step);
    let normal = normalize(cross(tangent_z, tangent_x));

    let world = vec3<f32>(x, y, z);

    var out: VsOut;
    out.clip_pos = params.view_proj * vec4<f32>(world, 1.0);
    out.world_pos = world;
    out.normal = normal;
    out.uv = vec2<f32>(u, age);
    out.amp = amp;
    return out;
}

fn sample_palette(t_in: f32, palette: u32) -> vec3<f32> {
    let t = clamp(t_in, 0.0, 1.0);

    if (palette == 1u) {
        // Magma Studio: Obsidian -> Violet -> Crimson -> Amber -> Hot White
        let c0 = vec3<f32>(0.025, 0.02, 0.06);
        let c1 = vec3<f32>(0.28, 0.05, 0.45);
        let c2 = vec3<f32>(0.88, 0.16, 0.24);
        let c3 = vec3<f32>(1.00, 0.65, 0.12);
        let c4 = vec3<f32>(1.00, 0.98, 0.88);
        if (t < 0.25) { return mix(c0, c1, t / 0.25); }
        if (t < 0.52) { return mix(c1, c2, (t - 0.25) / 0.27); }
        if (t < 0.80) { return mix(c2, c3, (t - 0.52) / 0.28); }
        return mix(c3, c4, (t - 0.80) / 0.20);
    } else if (palette == 2u) {
        // CRT Phosphor Green
        let c0 = vec3<f32>(0.01, 0.035, 0.02);
        let c1 = vec3<f32>(0.02, 0.25, 0.14);
        let c2 = vec3<f32>(0.08, 0.85, 0.42);
        let c3 = vec3<f32>(0.55, 1.00, 0.72);
        let c4 = vec3<f32>(0.94, 1.00, 0.96);
        if (t < 0.25) { return mix(c0, c1, t / 0.25); }
        if (t < 0.55) { return mix(c1, c2, (t - 0.25) / 0.30); }
        if (t < 0.82) { return mix(c2, c3, (t - 0.55) / 0.27); }
        return mix(c3, c4, (t - 0.82) / 0.18);
    } else if (palette == 3u) {
        // Ice Lab
        let c0 = vec3<f32>(0.015, 0.025, 0.06);
        let c1 = vec3<f32>(0.05, 0.18, 0.48);
        let c2 = vec3<f32>(0.08, 0.62, 0.92);
        let c3 = vec3<f32>(0.52, 0.92, 1.00);
        let c4 = vec3<f32>(0.96, 0.99, 1.00);
        if (t < 0.25) { return mix(c0, c1, t / 0.25); }
        if (t < 0.55) { return mix(c1, c2, (t - 0.25) / 0.30); }
        if (t < 0.82) { return mix(c2, c3, (t - 0.55) / 0.27); }
        return mix(c3, c4, (t - 0.82) / 0.18);
    }

    // Default (0): Vulkan Cyber Studio (Deep Sapphire -> Electric Teal -> Cyan -> Neon Emerald -> Magenta -> Hot White)
    let c0 = vec3<f32>(0.015, 0.025, 0.08);
    let c1 = vec3<f32>(0.02, 0.16, 0.44);
    let c2 = vec3<f32>(0.02, 0.56, 0.80);
    let c3 = vec3<f32>(0.12, 0.90, 0.62);
    let c4 = vec3<f32>(0.96, 0.24, 0.72);
    let c5 = vec3<f32>(1.00, 0.96, 1.00);
    if (t < 0.22) { return mix(c0, c1, t / 0.22); }
    if (t < 0.48) { return mix(c1, c2, (t - 0.22) / 0.26); }
    if (t < 0.72) { return mix(c2, c3, (t - 0.48) / 0.24); }
    if (t < 0.91) { return mix(c3, c4, (t - 0.72) / 0.19); }
    return mix(c4, c5, (t - 0.91) / 0.09);
}

fn freq_line(u: f32, target_hz: f32, f_min: f32, f_max: f32) -> f32 {
    let target_u = log(target_hz / f_min) / log(f_max / f_min);
    let d = abs(u - target_u);
    let fw = max(fwidth(u), 0.0015);
    return 1.0 - smoothstep(fw * 0.5, fw * 1.8, d);
}

@fragment
fn fs_waterfall(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let v = normalize(params.camera_pos.xyz - in.world_pos);

    let key_dir = normalize(vec3<f32>(-0.48, 0.82, 0.52));
    let fill_dir = normalize(vec3<f32>(0.68, 0.30, -0.45));

    let diff_key = max(dot(n, key_dir), 0.0);
    let diff_fill = max(dot(n, fill_dir), 0.0);

    let half_vec = normalize(key_dir + v);
    let spec = pow(max(dot(n, half_vec), 0.0), 48.0);
    let fresnel = pow(1.0 - max(dot(n, v), 0.0), 3.2);

    // Sculptural valley ambient occlusion prevents loud tracks from washing out into flat white
    let color_t = pow(clamp(in.amp, 0.0, 1.0), 1.32);
    let valley_ao = mix(0.38, 1.0, smoothstep(0.0, 0.42, in.amp));
    let base_col = sample_palette(color_t, params.grid_dims.w);
    let lit = base_col * (0.20 + 0.86 * diff_key + 0.20 * diff_fill) * valley_ao
        + vec3<f32>(0.42, 0.82, 1.0) * (spec * 0.38 + fresnel * 0.26 * (0.15 + color_t));

    // Studio grid lines: key mastering frequencies (100Hz, 1kHz, 10kHz) + subtle octaves + time slices
    let f_min = params.freq_range.x;
    let f_max = params.freq_range.y;
    var major_freq = 0.0;
    major_freq = max(major_freq, freq_line(in.uv.x, 100.0, f_min, f_max));
    major_freq = max(major_freq, freq_line(in.uv.x, 1000.0, f_min, f_max));
    major_freq = max(major_freq, freq_line(in.uv.x, 10000.0, f_min, f_max));

    var minor_freq = 0.0;
    minor_freq = max(minor_freq, freq_line(in.uv.x, 50.0, f_min, f_max));
    minor_freq = max(minor_freq, freq_line(in.uv.x, 250.0, f_min, f_max));
    minor_freq = max(minor_freq, freq_line(in.uv.x, 500.0, f_min, f_max));
    minor_freq = max(minor_freq, freq_line(in.uv.x, 2500.0, f_min, f_max));
    minor_freq = max(minor_freq, freq_line(in.uv.x, 5000.0, f_min, f_max));

    // Contour/wireframe slices along frequency & time
    let gx_grid = abs(fract(in.uv.x * 48.0) - 0.5);
    let gz_grid = abs(fract(in.uv.y * 36.0) - 0.5);
    let gx_line = 1.0 - smoothstep(0.014, 0.052, gx_grid);
    let gz_line = 1.0 - smoothstep(0.014, 0.052, gz_grid);
    let wire = max(gx_line, gz_line) * params.visual.z;

    // Live leading-edge highlight (newest slice glows crisp without overpowering)
    let live_edge = exp(-in.uv.y * 46.0) * (0.22 + color_t * 1.15);

    // HDR Emission reserved for true peaks (>0.65) so bloom catches loud ridges without washing out mids
    let hdr_peak = smoothstep(0.62, 0.98, in.amp) * params.visual.y;
    var col = lit * (1.0 + hdr_peak * 1.20);
    col += base_col * (wire * (0.24 + color_t * 0.72));
    col += vec3<f32>(0.25, 0.85, 1.0) * (major_freq * 0.36 + minor_freq * 0.12);
    col += (base_col + vec3<f32>(0.25, 0.75, 0.95)) * live_edge * 0.55;

    // Smooth studio horizon/edge fade
    let fade_back = smoothstep(1.0, 0.82, in.uv.y);
    let fade_side = smoothstep(0.0, 0.03, in.uv.x) * smoothstep(1.0, 0.97, in.uv.x);
    let bg = vec3<f32>(0.014, 0.018, 0.028);
    col = mix(bg, col, fade_back * (0.35 + 0.65 * fade_side));

    return vec4<f32>(col, 1.0);
}
