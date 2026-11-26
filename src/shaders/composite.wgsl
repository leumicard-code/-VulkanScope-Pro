struct CompositeParams {
    viewport: vec4<f32>,      // x = width, y = height, z = split_ratio (0.0..1.0), w = layout_mode (0=split, 1=scope, 2=waterfall)
    effects: vec4<f32>,       // x = bloom_strength, y = scanlines, z = graticule_alpha, w = rotate_ms
    crt_tint: vec4<f32>,      // rgb = scope phosphor tint, w = exposure
    top_bottom_pad: vec4<f32>,// x = top_px, y = bottom_px, z = blur_dir_x, w = blur_dir_y
};

@group(0) @binding(0) var<uniform> params: CompositeParams;
@group(0) @binding(1) var tex_a: texture_2d<f32>;
@group(0) @binding(2) var tex_b: texture_2d<f32>;
@group(0) @binding(3) var tex_bloom_a: texture_2d<f32>;
@group(0) @binding(4) var tex_bloom_b: texture_2d<f32>;
@group(0) @binding(5) var linear_sampler: sampler;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_quad(@builtin(vertex_index) vid: u32) -> VsOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>( 3.0,  1.0),
        vec2<f32>(-1.0,  1.0)
    );
    let p = pos[vid];
    var out: VsOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

// ---- Bloom Extract & Separable 9-tap Gaussian Blur ----
@fragment
fn fs_bloom_extract(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSampleLevel(tex_a, linear_sampler, in.uv, 0.0).rgb;
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Soft knee threshold extracts bright HDR peaks & CRT phosphor glow
    let knee = 0.35;
    let threshold = 0.42;
    let soft = clamp((luma - threshold + knee) / (2.0 * knee), 0.0, 1.0);
    let contrib = max(luma - threshold, 0.0) + soft * soft * knee * 0.5;
    let scale = contrib / max(luma, 1e-4);
    return vec4<f32>(c * scale, 1.0);
}

@fragment
fn fs_bloom_blur(in: VsOut) -> @location(0) vec4<f32> {
    let dir = vec2<f32>(params.top_bottom_pad.z, params.top_bottom_pad.w);
    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    var acc = textureSampleLevel(tex_a, linear_sampler, in.uv, 0.0).rgb * weights[0];
    for (var i = 1; i < 5; i = i + 1) {
        let off = dir * f32(i) * 1.45;
        acc += textureSampleLevel(tex_a, linear_sampler, in.uv + off, 0.0).rgb * weights[i];
        acc += textureSampleLevel(tex_a, linear_sampler, in.uv - off, 0.0).rgb * weights[i];
    }
    return vec4<f32>(acc, 1.0);
}

// ---- ACES Filmic Tone Mapping ----
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn crt_graticule(p: vec2<f32>, tint: vec3<f32>, rotate_ms: f32) -> vec3<f32> {
    // p in [-1, 1] square CRT screen space
    let r = length(p);
    var g = 0.0;

    // 8x8 oscilloscope divisions
    let grid_uv = abs(fract((p + 1.0) * 4.0) - 0.5);
    let major_x = 1.0 - smoothstep(0.004, 0.012, abs(p.x));
    let major_y = 1.0 - smoothstep(0.004, 0.012, abs(p.y));
    let sub_grid = max(
        1.0 - smoothstep(0.004, 0.014, 0.5 - grid_uv.x),
        1.0 - smoothstep(0.004, 0.014, 0.5 - grid_uv.y)
    );
    g += sub_grid * 0.13;
    g += max(major_x, major_y) * 0.28;

    // Fine crosshair tick marks along center X and Y axes
    let tick_x = (1.0 - smoothstep(0.006, 0.016, abs(fract(p.x * 20.0 + 0.5) - 0.5))) * step(abs(p.y), 0.025);
    let tick_y = (1.0 - smoothstep(0.006, 0.016, abs(fract(p.y * 20.0 + 0.5) - 0.5))) * step(abs(p.x), 0.025);
    g += max(tick_x, tick_y) * 0.32;

    // Outer unity circle + inner 0.5 circle
    let ring_outer = 1.0 - smoothstep(0.003, 0.009, abs(r - 0.82));
    let ring_inner = 1.0 - smoothstep(0.003, 0.009, abs(r - 0.41));
    g += ring_outer * 0.34 + ring_inner * 0.12;

    // 45-degree L / R mastering diagonals
    if (rotate_ms > 0.5) {
        let diag1 = 1.0 - smoothstep(0.003, 0.010, abs(p.x - p.y) * 0.7071);
        let diag2 = 1.0 - smoothstep(0.003, 0.010, abs(p.x + p.y) * 0.7071);
        g += max(diag1, diag2) * 0.18 * step(r, 0.86);
    }

    return mix(vec3<f32>(0.22, 0.45, 0.48), tint, 0.45) * g * params.effects.z;
}

fn render_crt_pane(local_px: vec2<f32>, pane_size: vec2<f32>) -> vec3<f32> {
    let side = min(pane_size.x, pane_size.y) * 0.92;
    let center = pane_size * 0.5;
    let p = (local_px - center) / max(side * 0.5, 1.0); // [-1, 1] inside square

    // Dark brushed studio bezel outside CRT glass
    let chassis_bg = vec3<f32>(0.014, 0.017, 0.024);

    // Rounded CRT glass tube mask
    let q = abs(p) - vec2<f32>(0.94, 0.94);
    let sdf_box = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - 0.06;
    if (sdf_box > 0.015) {
        return chassis_bg;
    }

    // Subtle glass curvature barrel distortion
    let r2 = dot(p, p);
    let curved_p = p * (1.0 + 0.035 * r2);
    let uv = vec2<f32>(curved_p.x * 0.5 + 0.5, 0.5 - curved_p.y * 0.5);

    // Deep CRT phosphor screen background with subtle green/cyan ambient glass glow
    let crt_bg = vec3<f32>(0.006, 0.014, 0.016) + params.crt_tint.rgb * (0.012 * exp(-r2 * 1.4));

    var col = crt_bg;
    if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
        let scope_hdr = textureSampleLevel(tex_a, linear_sampler, uv, 0.0).rgb;
        let scope_bloom = textureSampleLevel(tex_bloom_a, linear_sampler, uv, 0.0).rgb;
        col += scope_hdr + scope_bloom * params.effects.x;
        col += crt_graticule(curved_p, params.crt_tint.rgb, params.effects.w);
    }

    // Optional fine CRT scanlines
    if (params.effects.y > 0.01) {
        let scan = 0.5 + 0.5 * sin(local_px.y * 1.55);
        col *= 1.0 - params.effects.y * 0.22 * (1.0 - scan);
    }

    // Glass vignette + bezel rim highlight
    let vig = 1.0 - 0.28 * smoothstep(0.55, 1.35, r2);
    col *= vig;

    let bezel_edge = 1.0 - smoothstep(-0.012, 0.012, sdf_box);
    let rim = (1.0 - smoothstep(0.0, 0.016, abs(sdf_box))) * 0.14;
    return mix(chassis_bg, col, bezel_edge) + vec3<f32>(0.16, 0.32, 0.38) * rim;
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let w = params.viewport.x;
    let h = params.viewport.y;
    let top_pad = params.top_bottom_pad.x;
    let bot_pad = params.top_bottom_pad.y;
    let usable_h = max(h - top_pad - bot_pad, 1.0);

    let px = in.uv * vec2<f32>(w, h);
    let layout_mode = u32(params.viewport.w + 0.5);

    var hdr = vec3<f32>(0.014, 0.017, 0.025);

    if (px.y >= top_pad && px.y <= h - bot_pad) {
        let local_y = px.y - top_pad;

        if (layout_mode == 1u) {
            // Fullscreen CRT Phosphor XY Scope
            hdr = render_crt_pane(vec2<f32>(px.x, local_y), vec2<f32>(w, usable_h));
        } else if (layout_mode == 2u) {
            // Fullscreen 3D Waterfall
            let wf_uv = vec2<f32>(px.x / w, local_y / usable_h);
            let wf = textureSampleLevel(tex_b, linear_sampler, wf_uv, 0.0).rgb;
            let wf_bloom = textureSampleLevel(tex_bloom_b, linear_sampler, wf_uv, 0.0).rgb;
            hdr = wf + wf_bloom * params.effects.x;
        } else {
            // Split Studio: Left = CRT Scope, Right = 3D Waterfall
            let split_x = floor(w * params.viewport.z);
            if (px.x < split_x - 1.0) {
                hdr = render_crt_pane(vec2<f32>(px.x, local_y), vec2<f32>(split_x - 1.0, usable_h));
            } else if (px.x > split_x + 1.0) {
                let wf_w = max(w - split_x - 1.0, 1.0);
                let wf_uv = vec2<f32>((px.x - split_x - 1.0) / wf_w, local_y / usable_h);
                let wf = textureSampleLevel(tex_b, linear_sampler, wf_uv, 0.0).rgb;
                let wf_bloom = textureSampleLevel(tex_bloom_b, linear_sampler, wf_uv, 0.0).rgb;
                hdr = wf + wf_bloom * params.effects.x;
            } else {
                // Crisp metallic divider line
                hdr = vec3<f32>(0.09, 0.16, 0.22);
            }
        }
    }

    let mapped = aces_tonemap(hdr * params.crt_tint.w);
    let srgb = pow(max(mapped, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(srgb, 1.0);
}
