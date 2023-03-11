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
	padding: u32, // padding to make the struct 16 bytes -> support for more GPUs
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

fn compute_iterations_smooth(z0: vec2<f32>, c: vec2<f32>, max_iterations: u32) -> f32 {
	var zn = z0;
	var iteration = 0u;
	while (iteration < max_iterations && mod2(zn) < 4.0) {
		zn = compute_next(zn, c);
		iteration += 1u;
	}
	var m = mod2(zn);
	let smooth_iteration = f32(iteration) - max(0.0, log2(log2(m)));
	return smooth_iteration;
}

// Color gradient
fn colorGradient(x: f32) -> vec4<f32> {
	// Define the colors at each end of the gradient
	let blue = vec4<f32>(0.0, 0.0, 1.0, 1.0);
	let green = vec4<f32>(0.0, 1.0, 0.0, 1.0);
	let red = vec4<f32>(1.0, 0.0, 0.0, 1.0);

	// Determine where the color should be on the gradient
	let blueToGreen = mix(blue, green, x);
	let greenToRed = mix(green, red, x);

	// Return the final color
	return mix(blueToGreen, greenToRed, x);
}

// Fragment shader

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let x = in.clip_position.x * dimension.width_inv;
	let y = in.clip_position.y * dimension.height_inv;
	let z = vec2<f32>(x, y);
	let c = vec2<f32>(fractal_parameters.c_real, fractal_parameters.c_imag);
	let iterations = compute_iterations_smooth(z, c, fractal_parameters.max_iterations);
	return colorGradient(f32(iterations) / f32(fractal_parameters.max_iterations));
}