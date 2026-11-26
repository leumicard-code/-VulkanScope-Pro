use crate::dsp::FFT_SIZE;
use bytemuck::{Pod, Zeroable};
use egui_wgpu::ScreenDescriptor;
use glam::{Mat4, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

const WF_TEX_WIDTH: u32 = 512;
const WF_TEX_HEIGHT: u32 = 256;
const WF_GRID_X: u32 = 256;
const WF_GRID_Z: u32 = 192;
const SCOPE_RES: u32 = 1024;
const SCOPE_BLOOM_RES: u32 = 256;
const FFT_STAGES: u32 = 13; // 2^13 = 8192
const UNIFORM_ALIGN: u64 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayoutMode {
    SplitStudio,
    PhosphorOnly,
    WaterfallOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhosphorColor {
    TektronixCyan,
    LabGreen,
    AmberCrt,
    PlasmaViolet,
}

impl PhosphorColor {
    pub fn rgb(self) -> [f32; 3] {
        match self {
            Self::TektronixCyan => [0.10, 0.96, 0.88],
            Self::LabGreen => [0.14, 1.00, 0.36],
            Self::AmberCrt => [1.00, 0.68, 0.14],
            Self::PlasmaViolet => [0.76, 0.34, 1.00],
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::TektronixCyan => "Tektronix Cyan",
            Self::LabGreen => "P31 Lab Green",
            Self::AmberCrt => "Amber CRT",
            Self::PlasmaViolet => "Plasma Violet",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaterfallPalette {
    VulkanCyber = 0,
    MagmaStudio = 1,
    CrtPhosphor = 2,
    IceLab = 3,
}

impl WaterfallPalette {
    pub fn label(self) -> &'static str {
        match self {
            Self::VulkanCyber => "Vulkan Cyber",
            Self::MagmaStudio => "Magma Studio",
            Self::CrtPhosphor => "CRT Phosphor",
            Self::IceLab => "Ice Lab",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ScopeSettings {
    pub layout: LayoutMode,
    pub split_ratio: f32,

    // Phosphor XY CRT
    pub decay_tau: f32,
    pub beam_intensity: f32,
    pub beam_radius: f32,
    pub scope_gain: f32,
    pub rotate_ms: bool,
    pub phosphor_color: PhosphorColor,
    pub scanlines: f32,
    pub graticule_alpha: f32,

    // FFT & 3D Waterfall
    pub tilt_db_per_oct: f32,
    pub smoothing: f32,
    pub min_db: f32,
    pub max_db: f32,
    pub height_scale: f32,
    pub glow_boost: f32,
    pub wireframe_mix: f32,
    pub palette: WaterfallPalette,
    pub flow_toward: bool,
    pub frozen: bool,

    // Bloom & Exposure
    pub bloom_strength: f32,
    pub exposure: f32,

    // 3D Orbit Camera
    pub cam_yaw: f32,
    pub cam_pitch: f32,
    pub cam_distance: f32,
}

impl Default for ScopeSettings {
    fn default() -> Self {
        Self {
            layout: LayoutMode::SplitStudio,
            split_ratio: 0.40,

            decay_tau: 0.055,
            beam_intensity: 0.58,
            beam_radius: 0.011,
            scope_gain: 1.15,
            rotate_ms: true,
            phosphor_color: PhosphorColor::TektronixCyan,
            scanlines: 0.25,
            graticule_alpha: 0.85,

            tilt_db_per_oct: 3.0,
            smoothing: 0.52,
            min_db: -76.0,
            max_db: 0.0,
            height_scale: 2.55,
            glow_boost: 0.90,
            wireframe_mix: 0.30,
            palette: WaterfallPalette::VulkanCyber,
            flow_toward: true,
            frozen: false,

            bloom_strength: 0.72,
            exposure: 1.05,

            cam_yaw: -0.28,
            cam_pitch: 0.50,
            cam_distance: 8.6,
        }
    }
}

impl ScopeSettings {
    pub fn reset_camera(&mut self) {
        self.cam_yaw = -0.28;
        self.cam_pitch = 0.50;
        self.cam_distance = 8.6;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FftParamsUniform {
    sample_rate: f32,
    tilt_db_per_oct: f32,
    smoothing: f32,
    write_row: u32,
    min_db: f32,
    max_db: f32,
    stage: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PhosphorParamsUniform {
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
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WaterfallParamsUniform {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    grid_dims: [u32; 4],
    visual: [f32; 4],
    freq_range: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeParamsUniform {
    viewport: [f32; 4],
    effects: [f32; 4],
    crt_tint: [f32; 4],
    top_bottom_pad: [f32; 4],
}

struct SizedTargets {
    wf_hdr_view: wgpu::TextureView,
    wf_depth_view: wgpu::TextureView,
    wf_bloom_a_view: wgpu::TextureView,
    wf_bloom_b_view: wgpu::TextureView,
    wf_width: u32,
    wf_bloom_w: u32,
    wf_bloom_h: u32,
    bg_wf_extract: wgpu::BindGroup,
    bg_wf_blur_h: wgpu::BindGroup,
    bg_wf_blur_v: wgpu::BindGroup,
    bg_composite: wgpu::BindGroup,
}

pub struct VulkanRenderer {
    surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub adapter_name: String,
    pub backend_name: String,

    // Shared Audio / FFT GPU Buffers
    samples_buf: wgpu::Buffer,
    fft_params_buf: wgpu::Buffer,
    fft_window_pipeline: wgpu::ComputePipeline,
    fft_stage_pipeline: wgpu::ComputePipeline,
    fft_spectrum_pipeline: wgpu::ComputePipeline,
    fft_bg_a_to_b: wgpu::BindGroup,
    fft_bg_b_to_a: wgpu::BindGroup,
    write_row: u32,
    wf_advance_accum: f32,

    // Phosphor XY Scope
    phosphor_params_buf: wgpu::Buffer,
    phosphor_decay_pipeline: wgpu::RenderPipeline,
    phosphor_beam_pipeline: wgpu::RenderPipeline,
    phosphor_bg: wgpu::BindGroup,
    scope_hdr_view: wgpu::TextureView,
    scope_bloom_a_view: wgpu::TextureView,
    scope_bloom_b_view: wgpu::TextureView,
    bg_scope_extract: wgpu::BindGroup,
    bg_scope_blur_h: wgpu::BindGroup,
    bg_scope_blur_v: wgpu::BindGroup,

    // 3D Waterfall
    wf_params_buf: wgpu::Buffer,
    wf_index_buf: wgpu::Buffer,
    wf_index_count: u32,
    wf_pipeline: wgpu::RenderPipeline,
    wf_bg: wgpu::BindGroup,

    // Bloom & Composite
    comp_bgl: wgpu::BindGroupLayout,
    comp_params_buf: wgpu::Buffer,
    bloom_extract_pipeline: wgpu::RenderPipeline,
    bloom_blur_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    linear_sampler: wgpu::Sampler,
    sized: SizedTargets,

    // egui overlay renderer
    pub egui_renderer: egui_wgpu::Renderer,
    elapsed_sec: f32,
}

impl VulkanRenderer {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = wgpu::Backends::VULKAN | wgpu::Backends::GL;
        let instance = wgpu::Instance::new(instance_desc);

        let surface = instance
            .create_surface(window)
            .expect("failed to create Vulkan/wgpu surface");

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no compatible Vulkan/GPU adapter found");

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend_name = format!("{:?}", info.backend);
        log::info!("Selected GPU: {} ({})", adapter_name, backend_name);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("vulkanscope-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("failed to request Vulkan device");

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else if caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
            wgpu::PresentMode::Immediate
        } else {
            wgpu::PresentMode::Fifo
        };

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            color_space: wgpu::SurfaceColorSpace::default(),
            width,
            height,
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // ---- 1. Shared Audio Buffers & Waterfall Ring-Buffer Texture ----
        let samples_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("audio-samples-8192"),
            size: (FFT_SIZE * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let fft_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft-ping-a"),
            size: (FFT_SIZE * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let fft_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft-pong-b"),
            size: (FFT_SIZE * std::mem::size_of::<[f32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let prev_spec_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prev-spectrum-512"),
            size: (WF_TEX_WIDTH as usize * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let fft_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft-params-dynamic"),
            size: (FFT_STAGES as u64) * UNIFORM_ALIGN,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let wf_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("waterfall-ring-tex"),
            size: wgpu::Extent3d {
                width: WF_TEX_WIDTH,
                height: WF_TEX_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let wf_tex_view = wf_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // ---- 2. Compute FFT Pipelines ----
        let fft_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fft-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fft.wgsl").into()),
        });

        let fft_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fft-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<FftParamsUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let fft_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fft-pipeline-layout"),
            bind_group_layouts: &[Some(&fft_bgl)],
            immediate_size: 0,
        });

        let make_fft_cp = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&fft_pl),
                module: &fft_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let fft_window_pipeline = make_fft_cp("window_bitrev");
        let fft_stage_pipeline = make_fft_cp("fft_stage");
        let fft_spectrum_pipeline = make_fft_cp("spectrum_log_row");

        let make_fft_bg = |label: &str, src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &fft_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &fft_params_buf,
                            offset: 0,
                            size: wgpu::BufferSize::new(
                                std::mem::size_of::<FftParamsUniform>() as u64
                            ),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: samples_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dst.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: prev_spec_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(&wf_tex_view),
                    },
                ],
            })
        };
        let fft_bg_a_to_b = make_fft_bg("fft-bg-a-to-b", &fft_buf_a, &fft_buf_b);
        let fft_bg_b_to_a = make_fft_bg("fft-bg-b-to-a", &fft_buf_b, &fft_buf_a);

        // ---- 3. Phosphor XY Scope Pipelines ----
        let phosphor_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("phosphor-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/phosphor.wgsl").into()),
        });

        let phosphor_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("phosphor-params"),
            size: std::mem::size_of::<PhosphorParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let phosphor_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("phosphor-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let phosphor_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("phosphor-bg"),
            layout: &phosphor_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: phosphor_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: samples_buf.as_entire_binding(),
                },
            ],
        });

        let phosphor_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("phosphor-pl"),
            bind_group_layouts: &[Some(&phosphor_bgl)],
            immediate_size: 0,
        });

        // Multiplicative decay blend: out = 0 * src + dst * src = dst * decay
        let phosphor_decay_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("phosphor-decay"),
                layout: Some(&phosphor_pl),
                vertex: wgpu::VertexState {
                    module: &phosphor_shader,
                    entry_point: Some("vs_fullscreen"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &phosphor_shader,
                    entry_point: Some("fs_decay"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::Src,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        // Additive HDR electron-beam accumulation
        let phosphor_beam_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("phosphor-beam"),
                layout: Some(&phosphor_pl),
                vertex: wgpu::VertexState {
                    module: &phosphor_shader,
                    entry_point: Some("vs_beam"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &phosphor_shader,
                    entry_point: Some("fs_beam"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });

        let scope_hdr_view = create_hdr_view(&device, "scope-hdr", SCOPE_RES, SCOPE_RES);
        let scope_bloom_a_view =
            create_hdr_view(&device, "scope-bloom-a", SCOPE_BLOOM_RES, SCOPE_BLOOM_RES);
        let scope_bloom_b_view =
            create_hdr_view(&device, "scope-bloom-b", SCOPE_BLOOM_RES, SCOPE_BLOOM_RES);

        // ---- 4. 3D Spectrogram Waterfall Pipeline ----
        let wf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("waterfall-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/waterfall.wgsl").into()),
        });

        let wf_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("waterfall-params"),
            size: std::mem::size_of::<WaterfallParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut indices = Vec::with_capacity(((WF_GRID_X - 1) * (WF_GRID_Z - 1) * 6) as usize);
        for z in 0..(WF_GRID_Z - 1) {
            for x in 0..(WF_GRID_X - 1) {
                let i0 = z * WF_GRID_X + x;
                let i1 = i0 + 1;
                let i2 = i0 + WF_GRID_X;
                let i3 = i2 + 1;
                indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
            }
        }
        let wf_index_count = indices.len() as u32;
        let wf_index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("waterfall-indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let wf_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("waterfall-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let wf_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("waterfall-bg"),
            layout: &wf_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wf_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&wf_tex_view),
                },
            ],
        });

        let wf_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("waterfall-pl"),
            bind_group_layouts: &[Some(&wf_bgl)],
            immediate_size: 0,
        });

        let wf_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("waterfall-3d"),
            layout: Some(&wf_pl),
            vertex: wgpu::VertexState {
                module: &wf_shader,
                entry_point: Some("vs_waterfall"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &wf_shader,
                entry_point: Some("fs_waterfall"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ---- 5. Bloom & Studio Composite Pipelines ----
        let comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite-wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite.wgsl").into()),
        });

        // 5 dynamic slots: 0=composite, 1=scope_h, 2=scope_v, 3=wf_h, 4=wf_v
        let comp_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite-params-dynamic"),
            size: 5 * UNIFORM_ALIGN,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear-clamp-sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let comp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<CompositeParamsUniform>() as u64,
                        ),
                    },
                    count: None,
                },
                tex_entry(1),
                tex_entry(2),
                tex_entry(3),
                tex_entry(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let comp_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite-pl"),
            bind_group_layouts: &[Some(&comp_bgl)],
            immediate_size: 0,
        });

        let make_post_rp = |label: &str, entry: &str, format: wgpu::TextureFormat| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&comp_pl),
                vertex: wgpu::VertexState {
                    module: &comp_shader,
                    entry_point: Some("vs_quad"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &comp_shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let bloom_extract_pipeline =
            make_post_rp("bloom-extract", "fs_bloom_extract", wgpu::TextureFormat::Rgba16Float);
        let bloom_blur_pipeline =
            make_post_rp("bloom-blur", "fs_bloom_blur", wgpu::TextureFormat::Rgba16Float);
        let composite_pipeline = make_post_rp("composite", "fs_composite", surface_format);

        let bg_scope_extract = make_comp_bg(
            &device,
            &comp_bgl,
            &comp_params_buf,
            &scope_hdr_view,
            &scope_hdr_view,
            &scope_hdr_view,
            &scope_hdr_view,
            &linear_sampler,
        );
        let bg_scope_blur_h = make_comp_bg(
            &device,
            &comp_bgl,
            &comp_params_buf,
            &scope_bloom_a_view,
            &scope_hdr_view,
            &scope_hdr_view,
            &scope_hdr_view,
            &linear_sampler,
        );
        let bg_scope_blur_v = make_comp_bg(
            &device,
            &comp_bgl,
            &comp_params_buf,
            &scope_bloom_b_view,
            &scope_hdr_view,
            &scope_hdr_view,
            &scope_hdr_view,
            &linear_sampler,
        );

        let sized = create_sized_targets(
            &device,
            &comp_bgl,
            &comp_params_buf,
            &scope_hdr_view,
            &scope_bloom_a_view,
            &linear_sampler,
            width,
            height,
        );

        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            surface_format,
            egui_wgpu::RendererOptions::default(),
        );

        Self {
            surface,
            device,
            queue,
            config,
            adapter_name,
            backend_name,
            samples_buf,
            fft_params_buf,
            fft_window_pipeline,
            fft_stage_pipeline,
            fft_spectrum_pipeline,
            fft_bg_a_to_b,
            fft_bg_b_to_a,
            write_row: 0,
            wf_advance_accum: 0.0,
            phosphor_params_buf,
            phosphor_decay_pipeline,
            phosphor_beam_pipeline,
            phosphor_bg,
            scope_hdr_view,
            scope_bloom_a_view,
            scope_bloom_b_view,
            bg_scope_extract,
            bg_scope_blur_h,
            bg_scope_blur_v,
            wf_params_buf,
            wf_index_buf,
            wf_index_count,
            wf_pipeline,
            wf_bg,
            comp_bgl,
            comp_params_buf,
            bloom_extract_pipeline,
            bloom_blur_pipeline,
            composite_pipeline,
            linear_sampler,
            sized,
            egui_renderer,
            elapsed_sec: 0.0,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.sized = create_sized_targets(
            &self.device,
            &self.comp_bgl,
            &self.comp_params_buf,
            &self.scope_hdr_view,
            &self.scope_bloom_a_view,
            &self.linear_sampler,
            width,
            height,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_frame(
        &mut self,
        window: &Window,
        samples: &[[f32; 2]],
        fresh_samples: usize,
        sample_rate: u32,
        dt_sec: f32,
        settings: &ScopeSettings,
        top_pad_px: f32,
        bottom_pad_px: f32,
        egui_ctx: &egui::Context,
        egui_output: egui::FullOutput,
        pixels_per_point: f32,
    ) {
        self.elapsed_sec += dt_sec;

        // Upload 8192 stereo samples to GPU
        if !settings.frozen && samples.len() == FFT_SIZE {
            self.queue
                .write_buffer(&self.samples_buf, 0, bytemuck::cast_slice(samples));
        }

        // Advance waterfall ring buffer at ~60 rows/sec so time speed is consistent at 60Hz or 144Hz
        let mut run_fft = false;
        if !settings.frozen {
            self.wf_advance_accum += dt_sec;
            if self.wf_advance_accum >= (1.0 / 60.0) {
                self.wf_advance_accum = (self.wf_advance_accum - (1.0 / 60.0)).min(0.05);
                self.write_row = (self.write_row + 1) % WF_TEX_HEIGHT;
                run_fft = true;
            }
        }

        // Write dynamic FFT uniform slots (0..13)
        if run_fft {
            for stage in 0..FFT_STAGES {
                let u = FftParamsUniform {
                    sample_rate: sample_rate as f32,
                    tilt_db_per_oct: settings.tilt_db_per_oct,
                    smoothing: settings.smoothing,
                    write_row: self.write_row,
                    min_db: settings.min_db,
                    max_db: settings.max_db,
                    stage,
                    _pad: 0,
                };
                self.queue.write_buffer(
                    &self.fft_params_buf,
                    (stage as u64) * UNIFORM_ALIGN,
                    bytemuck::bytes_of(&u),
                );
            }
        }

        // Phosphor XY uniforms
        let decay = if settings.frozen {
            1.0
        } else {
            (-dt_sec / settings.decay_tau.max(0.008)).exp().clamp(0.45, 0.985)
        };
        let seg_count = if settings.frozen {
            0
        } else {
            fresh_samples.clamp(512, 2400).min(FFT_SIZE - 1) as u32
        };
        let sample_offset = (FFT_SIZE as u32).saturating_sub(seg_count + 1);
        let p_col = settings.phosphor_color.rgb();

        let phosphor_u = PhosphorParamsUniform {
            decay,
            beam_intensity: settings.beam_intensity,
            beam_radius: settings.beam_radius,
            rotate_ms: if settings.rotate_ms { 1.0 } else { 0.0 },
            color_r: p_col[0],
            color_g: p_col[1],
            color_b: p_col[2],
            gain: settings.scope_gain,
            sample_offset,
            segment_count: seg_count,
            _pad0: 0,
            _pad1: 0,
        };
        self.queue
            .write_buffer(&self.phosphor_params_buf, 0, bytemuck::bytes_of(&phosphor_u));

        // 3D Waterfall camera & uniforms
        let usable_h = (self.config.height as f32 - top_pad_px - bottom_pad_px).max(1.0);
        let wf_pane_w = match settings.layout {
            LayoutMode::SplitStudio => (self.config.width as f32 * (1.0 - settings.split_ratio)).max(1.0),
            LayoutMode::WaterfallOnly => self.config.width as f32,
            LayoutMode::PhosphorOnly => self.sized.wf_width as f32,
        };
        let aspect = (wf_pane_w / usable_h).max(0.2);

        let pitch = settings.cam_pitch.clamp(0.12, 1.48);
        let yaw = settings.cam_yaw;
        let dist = settings.cam_distance.clamp(4.2, 18.0);
        let target = Vec3::new(0.0, 0.72, 0.0);
        let cam_pos = target
            + Vec3::new(
                dist * pitch.cos() * yaw.sin(),
                dist * pitch.sin(),
                dist * pitch.cos() * yaw.cos(),
            );
        let view = look_at_rh(cam_pos, target, Vec3::Y);
        let proj = perspective_rh_wgpu(44.0f32.to_radians(), aspect, 0.1, 100.0);
        let view_proj = proj * view;

        let wf_u = WaterfallParamsUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_pos: [cam_pos.x, cam_pos.y, cam_pos.z, 1.0],
            grid_dims: [WF_GRID_X, WF_GRID_Z, self.write_row, settings.palette as u32],
            visual: [
                settings.height_scale,
                settings.glow_boost,
                settings.wireframe_mix,
                if settings.flow_toward { 1.0 } else { 0.0 },
            ],
            freq_range: [20.0, (sample_rate as f32 * 0.48).min(20_000.0), self.elapsed_sec, 0.0],
        };
        self.queue
            .write_buffer(&self.wf_params_buf, 0, bytemuck::bytes_of(&wf_u));

        // Write Composite + 4 Blur dynamic uniform slots
        let base_comp = CompositeParamsUniform {
            viewport: [
                self.config.width as f32,
                self.config.height as f32,
                settings.split_ratio,
                match settings.layout {
                    LayoutMode::SplitStudio => 0.0,
                    LayoutMode::PhosphorOnly => 1.0,
                    LayoutMode::WaterfallOnly => 2.0,
                },
            ],
            effects: [
                settings.bloom_strength,
                settings.scanlines,
                settings.graticule_alpha,
                if settings.rotate_ms { 1.0 } else { 0.0 },
            ],
            crt_tint: [p_col[0], p_col[1], p_col[2], settings.exposure],
            top_bottom_pad: [top_pad_px, bottom_pad_px, 0.0, 0.0],
        };

        let mut scope_h = base_comp;
        scope_h.top_bottom_pad[2] = 1.0 / (SCOPE_BLOOM_RES as f32);
        scope_h.top_bottom_pad[3] = 0.0;

        let mut scope_v = base_comp;
        scope_v.top_bottom_pad[2] = 0.0;
        scope_v.top_bottom_pad[3] = 1.0 / (SCOPE_BLOOM_RES as f32);

        let mut wf_h = base_comp;
        wf_h.top_bottom_pad[2] = 1.0 / (self.sized.wf_bloom_w as f32);
        wf_h.top_bottom_pad[3] = 0.0;

        let mut wf_v = base_comp;
        wf_v.top_bottom_pad[2] = 0.0;
        wf_v.top_bottom_pad[3] = 1.0 / (self.sized.wf_bloom_h as f32);

        self.queue.write_buffer(&self.comp_params_buf, 0 * UNIFORM_ALIGN, bytemuck::bytes_of(&base_comp));
        self.queue.write_buffer(&self.comp_params_buf, 1 * UNIFORM_ALIGN, bytemuck::bytes_of(&scope_h));
        self.queue.write_buffer(&self.comp_params_buf, 2 * UNIFORM_ALIGN, bytemuck::bytes_of(&scope_v));
        self.queue.write_buffer(&self.comp_params_buf, 3 * UNIFORM_ALIGN, bytemuck::bytes_of(&wf_h));
        self.queue.write_buffer(&self.comp_params_buf, 4 * UNIFORM_ALIGN, bytemuck::bytes_of(&wf_v));

        // Acquire next swapchain frame
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex)
            | wgpu::CurrentSurfaceTexture::Suboptimal(tex) => tex,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Validation => {
                log::error!("Surface validation error");
                return;
            }
        };
        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("vulkanscope-frame"),
            });

        // ---- PASS 1: 8192-Point Vulkan Compute FFT + Log Waterfall Slice ----
        if run_fft {
            {
                let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fft-window"),
                    timestamp_writes: None,
                });
                cp.set_pipeline(&self.fft_window_pipeline);
                cp.set_bind_group(0, &self.fft_bg_b_to_a, &[0]); // writes dst = buf_a
                cp.dispatch_workgroups((FFT_SIZE as u32) / 64, 1, 1);
            }
            for stage in 0..FFT_STAGES {
                let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fft-butterfly"),
                    timestamp_writes: None,
                });
                cp.set_pipeline(&self.fft_stage_pipeline);
                let bg = if stage % 2 == 0 {
                    &self.fft_bg_a_to_b
                } else {
                    &self.fft_bg_b_to_a
                };
                cp.set_bind_group(0, bg, &[stage * (UNIFORM_ALIGN as u32)]);
                cp.dispatch_workgroups((FFT_SIZE as u32 / 2) / 64, 1, 1);
            }
            {
                let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fft-log-waterfall"),
                    timestamp_writes: None,
                });
                cp.set_pipeline(&self.fft_spectrum_pipeline);
                // After stage 12 (even), final spectrum lives in buf_b -> src of bg_b_to_a
                cp.set_bind_group(0, &self.fft_bg_b_to_a, &[0]);
                cp.dispatch_workgroups(WF_TEX_WIDTH / 64, 1, 1);
            }
        }

        // ---- PASS 2: Phosphor XY Oscilloscope (Decay + Instanced Beam) ----
        if settings.layout != LayoutMode::WaterfallOnly {
            {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("phosphor-scope-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.scope_hdr_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if !settings.frozen {
                    rp.set_pipeline(&self.phosphor_decay_pipeline);
                    rp.set_bind_group(0, &self.phosphor_bg, &[]);
                    rp.draw(0..3, 0..1);
                }
                if seg_count > 0 {
                    rp.set_pipeline(&self.phosphor_beam_pipeline);
                    rp.set_bind_group(0, &self.phosphor_bg, &[]);
                    rp.draw(0..6, 0..seg_count);
                }
            }

            // Scope HDR Bloom chain
            run_post_pass(
                &mut encoder,
                "scope-bloom-extract",
                &self.scope_bloom_a_view,
                &self.bloom_extract_pipeline,
                &self.bg_scope_extract,
                0,
            );
            for _ in 0..2 {
                run_post_pass(
                    &mut encoder,
                    "scope-bloom-h",
                    &self.scope_bloom_b_view,
                    &self.bloom_blur_pipeline,
                    &self.bg_scope_blur_h,
                    1 * UNIFORM_ALIGN as u32,
                );
                run_post_pass(
                    &mut encoder,
                    "scope-bloom-v",
                    &self.scope_bloom_a_view,
                    &self.bloom_blur_pipeline,
                    &self.bg_scope_blur_v,
                    2 * UNIFORM_ALIGN as u32,
                );
            }
        }

        // ---- PASS 3: 3D Spectrogram Waterfall ----
        if settings.layout != LayoutMode::PhosphorOnly {
            {
                let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("waterfall-3d-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.sized.wf_hdr_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.014,
                                g: 0.018,
                                b: 0.028,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.sized.wf_depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Discard,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                rp.set_pipeline(&self.wf_pipeline);
                rp.set_bind_group(0, &self.wf_bg, &[]);
                rp.set_index_buffer(self.wf_index_buf.slice(..), wgpu::IndexFormat::Uint32);
                rp.draw_indexed(0..self.wf_index_count, 0, 0..1);
            }

            // Waterfall HDR Bloom chain
            run_post_pass(
                &mut encoder,
                "wf-bloom-extract",
                &self.sized.wf_bloom_a_view,
                &self.bloom_extract_pipeline,
                &self.sized.bg_wf_extract,
                0,
            );
            for _ in 0..2 {
                run_post_pass(
                    &mut encoder,
                    "wf-bloom-h",
                    &self.sized.wf_bloom_b_view,
                    &self.bloom_blur_pipeline,
                    &self.sized.bg_wf_blur_h,
                    3 * UNIFORM_ALIGN as u32,
                );
                run_post_pass(
                    &mut encoder,
                    "wf-bloom-v",
                    &self.sized.wf_bloom_a_view,
                    &self.bloom_blur_pipeline,
                    &self.sized.bg_wf_blur_v,
                    4 * UNIFORM_ALIGN as u32,
                );
            }
        }

        // ---- PASS 4: Composite + ACES Tone Map to Swapchain ----
        run_post_pass(
            &mut encoder,
            "studio-composite",
            &surface_view,
            &self.composite_pipeline,
            &self.sized.bg_composite,
            0,
        );

        // ---- PASS 5: egui Studio Mastering HUD ----
        let paint_jobs = egui_ctx.tessellate(egui_output.shapes, pixels_per_point);
        let screen_desc = ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point,
        };
        for (id, deltas) in &egui_output.textures_delta.set {
            for delta in deltas {
                self.egui_renderer
                    .update_texture(&self.device, &self.queue, *id, delta);
            }
        }
        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_desc,
        );
        {
            let rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui-hud-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            let mut rp_static = rp.forget_lifetime();
            self.egui_renderer
                .render(&mut rp_static, &paint_jobs, &screen_desc);
        }
        for id in &egui_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        window.pre_present_notify();
        self.queue.present(frame);
    }
}

fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Mat4 {
    let f = (center - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(f);
    Mat4::from_cols_array(&[
        s.x, u.x, -f.x, 0.0,
        s.y, u.y, -f.y, 0.0,
        s.z, u.z, -f.z, 0.0,
        -s.dot(eye), -u.dot(eye), f.dot(eye), 1.0,
    ])
}

/// Right-handed perspective projection mapping Z into wgpu/Vulkan clip range [0, 1].
fn perspective_rh_wgpu(fov_y_rad: f32, aspect: f32, z_near: f32, z_far: f32) -> Mat4 {
    let f = 1.0 / (0.5 * fov_y_rad).tan();
    let r = z_far / (z_near - z_far);
    Mat4::from_cols_array(&[
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, r, -1.0,
        0.0, 0.0, r * z_near, 0.0,
    ])
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn create_hdr_view(device: &wgpu::Device, label: &str, w: u32, h: u32) -> wgpu::TextureView {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

#[allow(clippy::too_many_arguments)]
fn make_comp_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params_buf: &wgpu::Buffer,
    t1: &wgpu::TextureView,
    t2: &wgpu::TextureView,
    t3: &wgpu::TextureView,
    t4: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("composite-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: params_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(
                        std::mem::size_of::<CompositeParamsUniform>() as u64,
                    ),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(t1),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(t2),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(t3),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(t4),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn create_sized_targets(
    device: &wgpu::Device,
    comp_bgl: &wgpu::BindGroupLayout,
    comp_params_buf: &wgpu::Buffer,
    scope_hdr_view: &wgpu::TextureView,
    scope_bloom_a_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    width: u32,
    height: u32,
) -> SizedTargets {
    let wf_w = width.max(16);
    let wf_h = height.max(16);
    let wf_bloom_w = (wf_w / 3).max(64);
    let wf_bloom_h = (wf_h / 3).max(64);

    let wf_hdr_view = create_hdr_view(device, "wf-hdr", wf_w, wf_h);
    let wf_bloom_a_view = create_hdr_view(device, "wf-bloom-a", wf_bloom_w, wf_bloom_h);
    let wf_bloom_b_view = create_hdr_view(device, "wf-bloom-b", wf_bloom_w, wf_bloom_h);

    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("wf-depth"),
        size: wgpu::Extent3d {
            width: wf_w,
            height: wf_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let wf_depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

    let bg_wf_extract = make_comp_bg(
        device,
        comp_bgl,
        comp_params_buf,
        &wf_hdr_view,
        &wf_hdr_view,
        &wf_hdr_view,
        &wf_hdr_view,
        sampler,
    );
    let bg_wf_blur_h = make_comp_bg(
        device,
        comp_bgl,
        comp_params_buf,
        &wf_bloom_a_view,
        &wf_hdr_view,
        &wf_hdr_view,
        &wf_hdr_view,
        sampler,
    );
    let bg_wf_blur_v = make_comp_bg(
        device,
        comp_bgl,
        comp_params_buf,
        &wf_bloom_b_view,
        &wf_hdr_view,
        &wf_hdr_view,
        &wf_hdr_view,
        sampler,
    );
    let bg_composite = make_comp_bg(
        device,
        comp_bgl,
        comp_params_buf,
        scope_hdr_view,
        &wf_hdr_view,
        scope_bloom_a_view,
        &wf_bloom_a_view,
        sampler,
    );

    SizedTargets {
        wf_hdr_view,
        wf_depth_view,
        wf_bloom_a_view,
        wf_bloom_b_view,
        wf_width: wf_w,
        wf_bloom_w,
        wf_bloom_h,
        bg_wf_extract,
        bg_wf_blur_h,
        bg_wf_blur_v,
        bg_composite,
    }
}

fn run_post_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bg: &wgpu::BindGroup,
    dynamic_offset: u32,
) {
    let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    rp.set_pipeline(pipeline);
    rp.set_bind_group(0, bg, &[dynamic_offset]);
    rp.draw(0..3, 0..1);
}
