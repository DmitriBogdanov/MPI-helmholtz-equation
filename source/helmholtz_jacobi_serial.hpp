#pragma once

#include <functional>
#include <omp.h>

#include "utils.hpp"

// ### Helholtz equation ###
//
//   { -u_xx - u_yy + k^2 u(x, y) = f(x, y),   in square region [0,L]x[0,L]
//   { u|_x=0 = boundary_left(y)
//   { u|_x=L = boundary_right(y)
//   { u|_y=0 = boundary_top(x)
//   { u|_y=L = boundary_bottom(x)
//   here
//   k - wave_number
//   f - right_part
//
// By taking a standard difference scheme aka 'cross' we get
//   alpha * y[i][j] - y[i-1][j] - y[i+1][j] - y[i][j-1] - y[i][j+1] = beta * f[i][j]
// where
//   alpha = (4 + h^2 k^2)
//   beta  = (h^2)
//
// On an even NxN grid:
//
//   u u u u u u u u u u
//   l 0 0 # 0 0 0 0 0 r
//   l 0 # # # 0 0 0 0 r  <- 'cross' difference scheme
//   l 0 0 # 0 0 0 0 0 r
//   l 0 0 0 0 0 0 0 0 r
//   l 0 0 0 0 0 0 0 0 r
//   l 0 0 0 0 0 0 0 0 r
//   l 0 0 0 0 0 0 0 0 r
//   l 0 0 0 0 0 0 0 0 r
//   b b b b b b b b b b
//
// By iterating 'cross' over all internal points of a grid we get a SLAE of N^2 variables,
// some of which are actually fixed due to boundaries being constant, this reduces SLAE size
// down to (internalN)^2 where 'internalN = N-2'
//
// This SLAE can be solved with any iterative method, here we use:
//
// ### Iterative Jacobi method  ###
//
//   A * x = f, 
//   (L + U) * x_k + D * x_k+1 = f, 
//   L + U = A - D,
//   A = L + D + U, L - upper triangular, D - diagonal, U - lower triangular
//   x_k+1 = D^{-1} (f - (A - D) x_k)
//   x_k+1 = y + C x_k,
//   y = D^{-1} * f, C = D^{-1} * (D - A)
//   -> Convergence condition: diagonal dominance
//
// Which in our case leads to following formula:
//   y[i][j] = alpha^-1 * (y0[i-1][j] + y0[i+1][j] + y0[i][j-1] + y0[i][j+1] + beta * f[i][j])
// where 'y0' is solution at the previous iteration
//
// This formula is iterated over all internal points of a grid to get a new 'y', effectively
// implementing Jacobi method without explicitly writing down SLAE matrix
//
// NOTE: Swapping pointers to 'y' and 'y0' each iterations allows us to avoid copying 'y'
//       into 'y0', which would be the naive way to implement 'y0 = y'
//
// NOTE: With matrix-like (i, j) indexation, 'i' corresponds to Y-axis, while 'j' corresponds to X-axis
//
// NOTE: Tabulating right part f(x,y) beforehand improves performance, but increases
//       memory consumption, which is often the limiting factor for this method
//
UniquePtrArray helholtz_jacobi_serial(T k, std::function<T(T, T)> f, T L, size_t N, T epsilon,
	std::function<T(T)> boundary_left, std::function<T(T)> boundary_right,
	std::function<T(T)> boundary_bot, std::function<T(T)> boundary_top
) {
	T h = T(1) / (N - 1);

	const T alpha = T(4) + sqr(h) * sqr(k);
	const T inverseAlpha = T(1) / alpha;
	const T beta = sqr(h);

	const int num_var = sqr(N);

	auto x = make_raw_array(num_var, 0); // initial guess in zero-vector
	auto x0 = make_raw_array(num_var);

	auto x_ptr = x.get();
	auto x0_ptr = x0.get();

	// Tabulate boundaries
	// Left
	//#pragma omp parallel for // No significant difference noticed, seems marginally faster without OMP
	for (int i = 0; i < N; ++i) {
		x_ptr[i * N + 0] = boundary_left(i * h);
		x0_ptr[i * N + 0] = boundary_left(i * h);
	}
	// Right
	//#pragma omp parallel for // No significant difference noticed, seems marginally faster without OMP
	for (int i = 0; i < N; ++i) {
		x_ptr[i * N + N - 1] = boundary_right(i * h);
		x0_ptr[i * N + N - 1] = boundary_right(i * h);
	}
	// Top
	//#pragma omp parallel for // No significant difference noticed, seems marginally faster without OMP
	for (int j = 1; j < N - 1; ++j) {
		x_ptr[0 * N + j] = boundary_top(j * h);
		x0_ptr[0 * N + j] = boundary_top(j * h);
	}
	// Bottom 
	//#pragma omp parallel for // No significant difference noticed, seems marginally faster without OMP
	for (int j = 1; j < N - 1; ++j) {
		x_ptr[(N - 1) * N + j] = boundary_bot(j * h);
		x0_ptr[(N - 1) * N + j] = boundary_bot(j * h);
	}

	T norm_diff_2 = 0;

	// Jacobi method
	do {

		// x0 = x
		std::swap(x_ptr, x0_ptr);

		// Internal loop
		for (int i = 1; i < N - 1; ++i) {
			for (int j = 1; j < N - 1; ++j) {
				const int indexIJ = i * N + j;

				x_ptr[indexIJ] = inverseAlpha * (
					x0_ptr[indexIJ - N] + // up
					x0_ptr[indexIJ + N] + // down
					x0_ptr[indexIJ - 1] + // left
					x0_ptr[indexIJ + 1] + // right
					beta * f(j * h, i * h)
					);
			}
		}

		// Stop condition
		norm_diff_2 = 0;
		for (int i = 0; i < num_var; ++i) norm_diff_2 += sqr(x_ptr[i] - x0_ptr[i]);

	} while (sqrt(norm_diff_2) > epsilon);

	// Return vector from last iteration
	if (x.get() == x_ptr) return x;
	else return x0;
}