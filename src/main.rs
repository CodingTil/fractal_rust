use num_complex::Complex;
use pixel_canvas::{Canvas, Color};
mod fractal;
use rand;
use rayon::prelude::*;
use std::time::SystemTime;
use noise::{NoiseFn, Perlin};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
   /// Anti aliasing factor
   #[arg(short, long, default_value_t = 4)]
   anti_aliasing: usize,

   /// Maximum iterations
   #[arg(short, long, default_value_t = 5000)]
   max_iterations: usize,

   /// Width of the image
   #[arg(short, long, default_value_t = 1920)]
   width: usize,

   /// Height of the image
   #[arg(short, long, default_value_t = 1080)]
   height: usize,
}

fn main() {
	let args: Args = Args::parse();

	let now = SystemTime::now();
	let perlin_re = Perlin::new(0);
	let perlin_im = Perlin::new(0);

	let canvas = Canvas::new(args.width, args.height)
		.title("Fractal")
		.hidpi(true)
		.show_ms(true);

	canvas.render(move |_, image| {
		let current_time = now.elapsed().unwrap().as_secs_f64() / 10.0;
		let constant: Complex<f64> = Complex {
			re: perlin_re.get([current_time, 0.0, 0.0]),
			im: perlin_im.get([0.0, current_time, 0.0]),
		};

		let height = args.height as f64;
		let width = args.width as f64;
		let height_scale: f64 = 1.0 / (height / 2.0);
		let width_scale: f64 = 1.0 / (width / 2.0);
		let scale = height_scale.max(width_scale);

		let pixel_iterations = image
			.par_chunks(args.width)
			.enumerate()
			.map(|(y, row)| {
				row.par_iter().enumerate().map(move |(x, _)| {
					let iterations = (0..args.anti_aliasing).into_par_iter().map(|_| {
						let x_offset = rand::random::<f64>() - 0.5;
						let y_offset = rand::random::<f64>() - 0.5;
						fractal::compute_iterations_smooth(
							Complex {
								re: ((x as f64 + x_offset) - width / 2.0) * scale,
								im: ((y as f64 + y_offset) - height / 2.0) * scale,
							},
							constant,
							args.max_iterations,
						)
					}).sum::<usize>() as f64 / args.anti_aliasing as f64;
					iterations
				}).collect::<Vec<_>>()
			}).collect::<Vec<_>>();
		let max_iteration_count = pixel_iterations
			.par_iter()
			.flatten()
			.max_by(|a, b| a.partial_cmp(b).unwrap())
			.unwrap();
		image.par_chunks_mut(width as usize).enumerate().for_each(|(y, row)| {
			row.par_iter_mut().enumerate().for_each(|(x, pixel)| {
				let iterations = pixel_iterations[y][x];
				let color = gradient_color(iterations as f64, *max_iteration_count);
				*pixel = color;
			});
		});
	});
}

fn gradient_color(iterations: f64, max_iterations: f64) -> Color {
	let gray = (255.0 * (iterations / max_iterations)) as u8;
	Color {
		r: gray,
		g: gray,
		b: gray,
	}
}
