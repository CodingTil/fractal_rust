struct VertexInput {
	@location(0) position: vec2<f32>,
};

struct VertexOutput {
	@builtin(position) clip_position: vec4<f32>,
};

struct DimensionUniform {
	width: u32,
	height: u32,
	width_inv: f32,
	height_inv: f32,
}
@group(0) @binding(0)
var<uniform> dimension: DimensionUniform;

struct FractalUniform {
	max_iterations: u32,
	c_real: f32,
	c_imag: f32,
	elapsed_time: f32,
}
@group(1) @binding(0)
var<uniform> fractal_parameters: FractalUniform;

@vertex
fn vs_main(
	model: VertexInput,
) -> VertexOutput {
	var out: VertexOutput;
	out.clip_position = vec4<f32>(model.position, 1.0, 1.0);
	return out;
}

// Fractal Functions
fn compute_next(z: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
	let x = z.x * z.x - z.y * z.y + c.x;
	let y = 2.0 * z.x * z.y + c.y;
	return vec2<f32>(x, y);
}

fn mod2(z: vec2<f32>) -> f32 {
	return z.x * z.x + z.y * z.y;
}

fn compute_iterations_smooth(c: vec2<f32>, max_iterations: u32) -> f32 {
	var zn = vec2<f32>(0.0, 0.0);
	var iteration = 0u;
	while (iteration < max_iterations && mod2(zn) < 4.0) {
		zn = compute_next(zn, c);
		iteration += 1u;
	}
	var m = mod2(zn);
	let smooth_iteration = f32(iteration);
	return smooth_iteration;
}

// Color gradient
fn colorGradient(factor: f32, uv_x: f32, uv_y: f32, time: f32) -> vec4<f32> {
	let f = clamp(pow(factor, 1.0 - factor * (time - 35.0) / 35.0), 0.0, 1.0);
	let r = mix(0.0, 1.0, f) * (uv_x * 0.5 + 0.5);
	let g = mix(0.0, 1.0, f) * (uv_y * 0.5 + 0.5);
	let b = mix(0.0, 1.0, f) * (-uv_x * 0.5 + 0.5);
	let a = 1.0 - exp(-factor * 20.0);
	return vec4<f32>(r, g, b, a);
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let mandelbrot_center = vec2<f32>(-0.5, 0.0);

	let zoom = pow(0.5, 15.0 * (0.5 - 0.5 * cos(0.255 * fractal_parameters.elapsed_time)));

	let uv_x = (2.0 * (f32(in.clip_position.x) - 0.5 * f32(dimension.width)) * dimension.width_inv) * f32(dimension.width) * dimension.height_inv;
	let uv_y = 2.0 * (f32(in.clip_position.y) - 0.5 * f32(dimension.height)) * dimension.height_inv;

	let lerp_factor = 1.0 - zoom;
	let target_c = vec2<f32>(mix(mandelbrot_center.x, fractal_parameters.c_real, lerp_factor),
							mix(mandelbrot_center.y, fractal_parameters.c_imag, lerp_factor));

	let c = vec2<f32>(target_c.x + uv_x * zoom, target_c.y + uv_y * zoom);

	let iterations = compute_iterations_smooth(c, fractal_parameters.max_iterations);
	return colorGradient(f32(iterations) / f32(fractal_parameters.max_iterations), uv_x, uv_y, fractal_parameters.elapsed_time);
}