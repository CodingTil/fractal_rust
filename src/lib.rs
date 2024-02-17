use std::iter;

use rand::seq::SliceRandom;

#[cfg(target_arch = "wasm32")]
use std::sync::Arc;
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;
#[cfg(target_arch = "wasm32")]
use winit::dpi::LogicalSize;

#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;
#[cfg(target_arch = "wasm32")]
use web_time::SystemTime;

use wgpu::{util::DeviceExt, StoreOp};
use winit::{
	dpi::PhysicalSize,
	event::*,
	event_loop::EventLoop,
	keyboard::{KeyCode, PhysicalKey},
	window::{Window, WindowBuilder},
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DimensionUniform {
	width: u32,
	height: u32,
	width_inv: f32,
	height_inv: f32,
}

impl DimensionUniform {
	fn new(width: u32, height: u32) -> Self {
		Self {
			width,
			height,
			width_inv: (1.0 / width as f64) as f32,
			height_inv: (1.0 / height as f64) as f32,
		}
	}

	fn update(&mut self, width: u32, height: u32) {
		self.width = width;
		self.height = height;
		self.width_inv = (1.0 / width as f64) as f32;
		self.height_inv = (1.0 / height as f64) as f32;
	}
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FractalUniform {
	max_iterations: u32,
	c_real: f32,
	c_imag: f32,
	elapsed_time: f32,
}

impl FractalUniform {
	fn new(max_iterations: u32) -> Self {
		let mut fractal_uniform = Self {
			max_iterations,
			c_real: 0.0,
			c_imag: 0.0,
			elapsed_time: 0.0,
		};
		fractal_uniform.reset();
		fractal_uniform
	}

	fn update(&mut self, total_elapsed: f32) {
		self.elapsed_time += total_elapsed * 0.5;

		if self.elapsed_time > 60.0 {
			self.reset()
		}
	}

	fn reset(&mut self) {
		self.elapsed_time = 0.0;
		let location_options: [[f32; 2]; 9] = [
			[0.281_717_93, 0.577_105_3],
			[-0.811_531_2, 0.201_429_58],
			[0.452_721_03, 0.396_494_27],
			[-0.745428, 0.113009],
			[-0.348_537_74, -0.606_592_24],
			[-0.348_314_94, -0.606_486_6],
			[-1.027705e-1, -9.475189e-1],
			[-0.5577, -0.6099],
			[-0.59990625, 0.429_070_32],
		];
		let random_location = location_options.choose(&mut rand::thread_rng()).unwrap();
		self.c_real = random_location[0];
		self.c_imag = random_location[1];
	}
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
	position: [f32; 2],
}

impl Vertex {
	const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];

	fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
		use std::mem;

		wgpu::VertexBufferLayout {
			array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
			step_mode: wgpu::VertexStepMode::Vertex,
			attributes: &Self::ATTRIBS,
		}
	}
}

const VERTICES: &[Vertex] = &[
	Vertex {
		position: [-1.0, -1.0],
	}, // A
	Vertex {
		position: [-1.0, 1.0],
	}, // B
	Vertex {
		position: [1.0, -1.0],
	}, // C
	Vertex {
		position: [1.0, 1.0],
	}, // D
];

const INDICES: &[u16] = &[0, 1, 2, 1, 2, 3];

struct State {
	surface: wgpu::Surface,
	device: wgpu::Device,
	queue: wgpu::Queue,
	config: wgpu::SurfaceConfiguration,
	size: winit::dpi::PhysicalSize<u32>,
	render_pipeline: wgpu::RenderPipeline,
	vertex_buffer: wgpu::Buffer,
	index_buffer: wgpu::Buffer,
	num_indices: u32,
	dimension_uniform: DimensionUniform,
	dimension_uniform_buffer: wgpu::Buffer,
	dimension_uniform_bind_group: wgpu::BindGroup,
	fractal_uniform: FractalUniform,
	fractal_uniform_buffer: wgpu::Buffer,
	fractal_uniform_bind_group: wgpu::BindGroup,
	last_time: SystemTime,
	window: Window,
}

#[cfg(not(target_arch = "wasm32"))]
impl State {
	async fn new(window: Window) -> Self {
		let instance = State::create_instance();
		let (surface, size) = State::create_surface(&instance, &window);
		State::init(instance, window, surface, size).await
	}

	fn create_surface(
		instance: &wgpu::Instance,
		window: &Window,
	) -> (wgpu::Surface, PhysicalSize<u32>) {
		let size = window.inner_size();

		// # Safety
		//
		// The surface needs to live as long as the window that created it.
		// State owns the window so this should be safe.
		let surface = unsafe { instance.create_surface(&window) }.unwrap();

		(surface, size)
	}
}

#[cfg(target_arch = "wasm32")]
impl State {
	async fn new(canvas_arc: Arc<HtmlCanvasElement>, event_loop: &EventLoop<()>) -> Self {
		let canvas: HtmlCanvasElement = Arc::try_unwrap(canvas_arc).unwrap();
		let (width, height) = (canvas.client_width(), canvas.client_height());
		let instance = State::create_instance();
		let (surface, size) = State::create_surface(&instance, &canvas);
		use winit::platform::web::WindowBuilderExtWebSys;
		let window = WindowBuilder::new()
			.with_canvas(Some(canvas))
			.build(&event_loop)
			.map(|w| {
				// Set initial view port -- ** This isn't what we want! **
				// We want the canvas to always fit to the document.
				let _ = w.request_inner_size(LogicalSize::new(width, height));
				w
			})
			.expect("Could not build window");
		State::init(instance, window, surface, size).await
	}

	fn create_surface(
		instance: &wgpu::Instance,
		canvas: &HtmlCanvasElement,
	) -> (wgpu::Surface, PhysicalSize<u32>) {
		let size = PhysicalSize::new(canvas.client_width() as u32, canvas.client_height() as u32);

		let surface = instance.create_surface_from_canvas(canvas.clone()).unwrap();

		(surface, size)
	}
}

impl State {
	fn create_instance() -> wgpu::Instance {
		// The instance is a handle to our GPU
		// BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
		wgpu::Instance::new(wgpu::InstanceDescriptor {
			backends: wgpu::Backends::all(),
			dx12_shader_compiler: Default::default(),
			flags: Default::default(),
			gles_minor_version: Default::default(),
		})
	}

	async fn init(
		instance: wgpu::Instance,
		window: Window,
		surface: wgpu::Surface,
		size: PhysicalSize<u32>,
	) -> Self {
		let adapter = instance
			.request_adapter(&wgpu::RequestAdapterOptions {
				power_preference: wgpu::PowerPreference::default(),
				compatible_surface: Some(&surface),
				force_fallback_adapter: false,
			})
			.await
			.unwrap();

		let (device, queue) = adapter
			.request_device(
				&wgpu::DeviceDescriptor {
					label: None,
					features: wgpu::Features::empty(),
					// WebGL doesn't support all of wgpu's features, so if
					// we're building for the web we'll have to disable some.
					limits: if cfg!(target_arch = "wasm32") {
						wgpu::Limits::downlevel_webgl2_defaults()
					} else {
						wgpu::Limits::default()
					},
				},
				None, // Trace path
			)
			.await
			.unwrap();

		let surface_caps = surface.get_capabilities(&adapter);
		// Shader code in this tutorial assumes an Srgb surface texture. Using a different
		// one will result all the colors comming out darker. If you want to support non
		// Srgb surfaces, you'll need to account for that when drawing to the frame.
		let surface_format = surface_caps
			.formats
			.iter()
			.copied()
			.find(|f| f.is_srgb())
			.unwrap_or(surface_caps.formats[0]);
		let config = wgpu::SurfaceConfiguration {
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			format: surface_format,
			width: size.width,
			height: size.height,
			present_mode: surface_caps.present_modes[0],
			alpha_mode: surface_caps.alpha_modes[0],
			view_formats: vec![],
		};
		surface.configure(&device, &config);

		let dimension_uniform = DimensionUniform::new(size.width, size.height);
		let dimension_uniform_buffer =
			device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
				label: Some("dimension_uniform_buffer"),
				contents: bytemuck::cast_slice(&[dimension_uniform]),
				usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			});
		let dimension_uniform_bind_group_layout =
			device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
				label: Some("dimension_uniform_bind_group_layout"),
				entries: &[wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: None,
					},
					count: None,
				}],
			});
		let dimension_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("dimension_uniform_bind_group"),
			layout: &dimension_uniform_bind_group_layout,
			entries: &[wgpu::BindGroupEntry {
				binding: 0,
				resource: dimension_uniform_buffer.as_entire_binding(),
			}],
		});
		let fractal_uniform = FractalUniform::new(10000);
		let fractal_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("fractal_uniform_buffer"),
			contents: bytemuck::cast_slice(&[fractal_uniform]),
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
		});
		let fractal_uniform_bind_group_layout =
			device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
				label: Some("fractal_uniform_bind_group_layout"),
				entries: &[wgpu::BindGroupLayoutEntry {
					binding: 0,
					visibility: wgpu::ShaderStages::FRAGMENT,
					ty: wgpu::BindingType::Buffer {
						ty: wgpu::BufferBindingType::Uniform,
						has_dynamic_offset: false,
						min_binding_size: None,
					},
					count: None,
				}],
			});
		let fractal_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("fractal_uniform_bind_group"),
			layout: &fractal_uniform_bind_group_layout,
			entries: &[wgpu::BindGroupEntry {
				binding: 0,
				resource: fractal_uniform_buffer.as_entire_binding(),
			}],
		});

		let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

		let render_pipeline_layout =
			device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
				label: Some("Render Pipeline Layout"),
				bind_group_layouts: &[
					&dimension_uniform_bind_group_layout,
					&fractal_uniform_bind_group_layout,
				],
				push_constant_ranges: &[],
			});

		let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
			label: Some("Render Pipeline"),
			layout: Some(&render_pipeline_layout),
			vertex: wgpu::VertexState {
				module: &shader,
				entry_point: "vs_main",
				buffers: &[Vertex::desc()],
			},
			fragment: Some(wgpu::FragmentState {
				module: &shader,
				entry_point: "fs_main",
				targets: &[Some(wgpu::ColorTargetState {
					format: config.format,
					blend: Some(wgpu::BlendState {
						color: wgpu::BlendComponent::REPLACE,
						alpha: wgpu::BlendComponent::REPLACE,
					}),
					write_mask: wgpu::ColorWrites::ALL,
				})],
			}),
			primitive: wgpu::PrimitiveState {
				topology: wgpu::PrimitiveTopology::TriangleList,
				strip_index_format: None,
				front_face: wgpu::FrontFace::Ccw,
				cull_mode: None,
				// Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
				// or Features::POLYGON_MODE_POINT
				polygon_mode: wgpu::PolygonMode::Fill,
				// Requires Features::DEPTH_CLIP_CONTROL
				unclipped_depth: false,
				// Requires Features::CONSERVATIVE_RASTERIZATION
				conservative: false,
			},
			depth_stencil: None,
			multisample: wgpu::MultisampleState {
				count: 1,
				mask: !0,
				alpha_to_coverage_enabled: false,
			},
			// If the pipeline will be used with a multiview render pass, this
			// indicates how many array layers the attachments will have.
			multiview: None,
		});

		let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("Vertex Buffer"),
			contents: bytemuck::cast_slice(VERTICES),
			usage: wgpu::BufferUsages::VERTEX,
		});
		let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
			label: Some("Index Buffer"),
			contents: bytemuck::cast_slice(INDICES),
			usage: wgpu::BufferUsages::INDEX,
		});
		let num_indices = INDICES.len() as u32;

		let last_time = SystemTime::now();

		Self {
			surface,
			device,
			queue,
			config,
			size,
			render_pipeline,
			vertex_buffer,
			index_buffer,
			num_indices,
			dimension_uniform,
			dimension_uniform_buffer,
			dimension_uniform_bind_group,
			fractal_uniform,
			fractal_uniform_buffer,
			fractal_uniform_bind_group,
			last_time,
			window,
		}
	}

	pub fn window(&self) -> &Window {
		&self.window
	}

	pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
		if new_size.width > 0 && new_size.height > 0 {
			self.size = new_size;
			self.config.width = new_size.width;
			self.config.height = new_size.height;
			self.surface.configure(&self.device, &self.config);
			self.dimension_uniform
				.update(new_size.width, new_size.height);
			self.queue.write_buffer(
				&self.dimension_uniform_buffer,
				0,
				bytemuck::cast_slice(&[self.dimension_uniform]),
			);
		}
	}

	#[allow(unused_variables)]
	fn input(&mut self, event: &WindowEvent) -> bool {
		false
	}

	fn update(&mut self) {
		let now = SystemTime::now();
		let total_elapsed = now.duration_since(self.last_time).unwrap().as_secs_f32();
		self.last_time = now;
		self.fractal_uniform.update(total_elapsed);
		self.queue.write_buffer(
			&self.fractal_uniform_buffer,
			0,
			bytemuck::cast_slice(&[self.fractal_uniform]),
		);
	}

	fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
		let output = self.surface.get_current_texture()?;
		let view = output
			.texture
			.create_view(&wgpu::TextureViewDescriptor::default());

		let mut encoder = self
			.device
			.create_command_encoder(&wgpu::CommandEncoderDescriptor {
				label: Some("Render Encoder"),
			});

		{
			let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("Render Pass"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: &view,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color {
							r: 1.0,
							g: 1.0,
							b: 1.0,
							a: 1.0,
						}),
						store: StoreOp::Store,
					},
				})],
				depth_stencil_attachment: None,
				timestamp_writes: None,
				occlusion_query_set: None,
			});

			render_pass.set_pipeline(&self.render_pipeline);
			render_pass.set_bind_group(0, &self.dimension_uniform_bind_group, &[]);
			render_pass.set_bind_group(1, &self.fractal_uniform_bind_group, &[]);
			render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
			render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
			render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
		}

		self.queue.submit(iter::once(encoder.finish()));
		output.present();

		Ok(())
	}
}

fn init_logging() {
	cfg_if::cfg_if! {
		if #[cfg(target_arch = "wasm32")] {
			std::panic::set_hook(Box::new(console_error_panic_hook::hook));
			console_log::init_with_level(log::Level::Warn).expect("Could't initialize logger");
		} else {
			env_logger::init();
		}
	}
}

#[cfg(not(target_arch = "wasm32"))]
fn create_window(event_loop: &EventLoop<()>) -> Window {
	WindowBuilder::new().build(event_loop).unwrap()
}

fn run_loop(mut state: State, event_loop: EventLoop<()>) {
	let result = event_loop.run(move |event, control_flow| {
		match event {
			Event::WindowEvent {
				ref event,
				window_id,
			} if window_id == state.window().id() => {
				if !state.input(event) {
					match event {
						WindowEvent::CloseRequested
						| WindowEvent::KeyboardInput {
							event:
								KeyEvent {
									state: ElementState::Pressed,
									physical_key: PhysicalKey::Code(KeyCode::Escape),
									..
								},
							..
						} => control_flow.exit(),
						WindowEvent::Resized(physical_size) => {
							state.resize(*physical_size);
							state.window().request_redraw();
						}
						WindowEvent::RedrawRequested if window_id == state.window().id() => {
							state.update();
							match state.render() {
								Ok(_) => {}
								// Reconfigure the surface if it's lost or outdated
								Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
									state.resize(state.size)
								}
								// The system is out of memory, we should probably quit
								Err(wgpu::SurfaceError::OutOfMemory) => control_flow.exit(),
								// We're ignoring timeouts
								Err(wgpu::SurfaceError::Timeout) => {
									log::warn!("Surface timeout")
								}
							}
						}
						_ => {}
					}
				}
			}
			Event::AboutToWait => {
				state.window().request_redraw();
			}
			_ => {}
		}
	});
	result.unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn run() {
	init_logging();
	let event_loop = EventLoop::new().unwrap();
	let window = create_window(&event_loop);
	let state = State::new(window).await;
	run_loop(state, event_loop)
}

#[cfg(target_arch = "wasm32")]
pub async fn run() {
	init_logging();
	let event_loop = EventLoop::new().unwrap();
	use wasm_bindgen::JsCast;
	let canvas = web_sys::window()
		.and_then(|w| w.document())
		.and_then(|d| d.get_element_by_id("wasm-block"))
		.map(|e| e.unchecked_into::<HtmlCanvasElement>())
		.expect("Canvas not found");
	let state = State::new(Arc::new(canvas), &event_loop).await;
	run_loop(state, event_loop)
}

#[cfg(target_arch = "wasm32")]
pub async fn run_with_canvas(canvas_arc: Arc<HtmlCanvasElement>) {
	init_logging();
	let event_loop = EventLoop::new().unwrap();
	let state = State::new(canvas_arc, &event_loop).await;
	run_loop(state, event_loop)
}
