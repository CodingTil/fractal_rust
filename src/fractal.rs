use num_complex::Complex;

fn compute_next(z: Complex<f64>, c: Complex<f64>) -> Complex<f64> {
	z * z + c
}

fn mod2(z: Complex<f64>) -> f64 {
	z.re * z.re + z.im * z.im
}

#[allow(dead_code)]
pub fn compute_iterations(z0: Complex<f64>, c: Complex<f64>, max_iterations: usize) -> usize {
	let mut zn = z0;
	let mut iteration = 0;
	while iteration < max_iterations && mod2(zn) < 4.0 {
		zn = compute_next(zn, c);
		iteration += 1;
	}
	iteration
}

#[allow(dead_code)]
pub fn compute_iterations_smooth(
	z0: Complex<f64>,
	c: Complex<f64>,
	max_iterations: usize,
) -> usize {
	let mut zn = z0;
	let mut iteration = 0;
	while iteration < max_iterations && mod2(zn) < 4.0 {
		zn = compute_next(zn, c);
		iteration += 1;
	}
	let m = mod2(zn).sqrt();
	let smooth_iteration = iteration as f64 - 0_f64.max(m.log2().log2());
	smooth_iteration as usize
}
