#pragma once

#include <functional>
#include <mpi.h>

#include "utils.hpp"

#define MPI_T MPI_DOUBLE
#define MASTER_PROCESS 0

#define RANKPRINT outstream << "[" << MPI_rank << "/" << MPI_size << "]: " /// TEMP


// MPI Communication types
inline void MPI_sync_rows_1(T *x_ptr, int MPI_rank, int MPI_size, int rowsHeldByProcess, int N) {
	const auto rowTop = x_ptr;
	const auto rowBot = x_ptr + (rowsHeldByProcess - 1) * N;
	const auto rowUpper = rowTop + N;
	const auto rowLower = rowBot - N;

	// MPI_Recv() 1  upper row from the process {MPI_rank - 1}
	if (MPI_rank > 0) MPI_Recv(
		rowTop, N, MPI_T, MPI_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE
	);

	// MPI_Send() 1 bottom row to the process {MPI_rank + 1}
	if (MPI_rank < MPI_size - 1) MPI_Send(
		rowLower, N, MPI_T, MPI_rank + 1, 0, MPI_COMM_WORLD
	);

	// MPI_Recv() 1 bottom row from the process{ MPI_rank + 1 }
	if (MPI_rank < MPI_size - 1) MPI_Recv(
		rowBot, N, MPI_T, MPI_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE
	);

	// MPI_Send() 1  upper row to the process {MPI_rank - 1}
	if (MPI_rank > 0) MPI_Send(
		rowUpper, N, MPI_T, MPI_rank - 1, 0, MPI_COMM_WORLD
	);
}

inline void MPI_sync_rows_2(T *x_ptr, int MPI_rank, int MPI_size, int rowsHeldByProcess, int N) {
	const auto rowTop = x_ptr;
	const auto rowBot = x_ptr + (rowsHeldByProcess - 1) * N;
	const auto rowUpper = rowTop + N;
	const auto rowLower = rowBot - N;

	// First rank
	if (MPI_rank == 0) MPI_Sendrecv(
		rowLower, N, MPI_T, MPI_rank + 1, 0,
		rowBot, N, MPI_T, MPI_rank + 1, 0,
		MPI_COMM_WORLD, MPI_STATUSES_IGNORE
	);
	// Last rank
	else if (MPI_rank == MPI_size - 1) MPI_Sendrecv(
		rowUpper, N, MPI_T, MPI_rank - 1, 0,
		rowTop, N, MPI_T, MPI_rank - 1, 0,
		MPI_COMM_WORLD, MPI_STATUSES_IGNORE
	);
	// Middle ranks
	else {
		MPI_Sendrecv(
			rowLower, N, MPI_T, MPI_rank + 1, 0,
			rowTop, N, MPI_T, MPI_rank - 1, 0,
			MPI_COMM_WORLD, MPI_STATUSES_IGNORE
		);

		MPI_Sendrecv(
			rowUpper, N, MPI_T, MPI_rank - 1, 0,
			rowBot, N, MPI_T, MPI_rank + 1, 0,
			MPI_COMM_WORLD, MPI_STATUSES_IGNORE
		);
	}
}


// ### Helholtz equation ###
//
//   { -u_xx - u_yy + k^2 u(x, y) = f(x, y),   in square region [0,L]x[0,L]
//   { u|_x=0 = boundary_left(y)
//   { u|_x=L = boundary_right(y)
//   { u|_y=0 = boundary_top(y)
//   { u|_y=L = boundary_bottom(y)
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
// ### MPI parallelization ###
//
// Example for 'N = 10'
// => 'internalN = N - 2 = 8'
//
// Computation is divided evenly by rows, each rank iterates over 'rowsProcessedByRank = internalN / MPI_size',
// which means it needs to hold 'rowsHeldByRank = rowsProcessedByRank + 2' rows in its memory
//
//   u u u u u u u u u u    | rank 0
//   l 0 0 0 0 0 0 0 0 r    <
//   l 0 0 0 0 0 0 0 0 r    <      | rank 1
//   l 0 0 0 0 0 0 0 0 r    |      <
//   l 0 0 0 0 0 0 0 0 r           <      | rank 2
//   l 0 0 0 0 0 0 0 0 r           |      <
//   l 0 0 0 0 0 0 0 0 r                  <      | rank 3
//   l 0 0 0 0 0 0 0 0 r                  |      <
//   l 0 0 0 0 0 0 0 0 r                         <
//   b b b b b b b b b b                         |
//
// Every rank iterates over its rows and then sends its upper and lower processed rows to
// ranks above and below, syncronizing the spots where held block intersect with values
// from new iteration. This action essentialy gets matrix synced between all rankes and
// we can proceed to next iteration.
//
// Each rank computes '||x - x_0||' on its rows and sends resulting sums to master, which
// computes total '||x - x_0||' and checks stop condition '||x - x_0|| < epsilon', then
// master sends stop condition state to all other ranks so they exit the loop if needed
//
// NOTE: 'MPI_comminication_type' parameter selects one of following communication methods:
// 1) MPI_Send() + MPI_Reacv()
// 2) MPI_Sendrecv()
// 3) MPI_Isend() + MPI_Irecv()
//
inline UniquePtrArray helholtz_jacobi(T k, std::function<T(T, T)> f, T L, size_t N, T epsilon,
	std::function<T(T)> BC_left, std::function<T(T)> BC_right,
	std::function<T(T)> BC_bot, std::function<T(T)> BC_top,
	int MPI_rank, int MPI_size, int MPI_comminication_type
) {
	T h = T(1) / (N - 1);

	const T alpha = T(4) + sqr(h) * sqr(k);
	const T inverseAlpha = T(1) / alpha;
	const T beta = sqr(h);

	const int internalN = N - 2;

	const int rowsProcessedByRank = internalN / MPI_size;
	const int rowsHeldByRank = rowsProcessedByRank + 2;

	// Each process holds (internalN / MPI_size + 2) rows of 'X' and 'X0'
	auto x = make_raw_array(rowsHeldByRank * N, 0); // initial guess in zero-vector
	auto x0 = make_raw_array(rowsHeldByRank * N);

	auto x_ptr = x.get();
	auto x0_ptr = x0.get();

	// Tabulate boundaries
	// Top (held by process {0})
	if (MPI_rank == 0) for (int j = 1; j < N - 1; ++j) {
		x_ptr[0 * N + j] = BC_top(j * h);
		x0_ptr[0 * N + j] = BC_top(j * h);
	}
	// Bottom (held by process {MPI_size - 1})
	else if (MPI_rank == MPI_size - 1) for (int j = 1; j < N - 1; ++j) {
		x_ptr[(rowsHeldByRank - 1) * N + j] = BC_bot(j * h);
		x0_ptr[(rowsHeldByRank - 1) * N + j] = BC_bot(j * h);
	}
	// Left + Right (held by any other process)
	else for (int i = 0; i < rowsHeldByRank; ++i) {
		x_ptr[i * N + 0] = BC_left(i * h);
		x0_ptr[i * N + 0] = BC_left(i * h);

		x_ptr[i * N + N - 1] = BC_right(i * h);
		x0_ptr[i * N + N - 1] = BC_right(i * h);
	}
	
	int escapeFlag; // MPI doesn't know bools exist

	int count = 0; /// TEMP

	// Jacobi method
	do {
		if (++count > 30) break; /// TEMP

		RANKPRINT << "iteration - " << count << std::endl; /// TEMP

		// x0 = x
		std::swap(x_ptr, x0_ptr);

		// Internal loop
		for (int i = 1; i < rowsHeldByRank - 1; ++i) {
			for (int j = 1; j < N - 1; ++j) {
				const int indexIJ = i * N + j;

				const T gridX = j * h;
				const T gridY = (MPI_rank * rowsProcessedByRank + i) * h; // gotta account for current block position

				x_ptr[indexIJ] = inverseAlpha * (
					x0_ptr[indexIJ - N] + // up
					x0_ptr[indexIJ + N] + // down
					x0_ptr[indexIJ - 1] + // left
					x0_ptr[indexIJ + 1] + // right
					beta * f(gridX, gridY)
						// NOTE: tabulating f(x,y) beforehand improves performance, but increases
						// memory consumption which is often the limiting factor for this method
					);
			}
		}

		// Syncronize state
		switch (MPI_comminication_type) {
		case 1:
			MPI_sync_rows_1(x_ptr, MPI_rank, MPI_size, rowsHeldByRank, N);
			break;
		case 2:
			MPI_sync_rows_2(x_ptr, MPI_rank, MPI_size, rowsHeldByRank, N);
			break;
		case 3:
			break;
		default:
			exit_with_error("Unknown 'MPI_comminication_type' in Jacobi method");
		}

		// Get stop condition state from master and break out if needed
		if (MPI_rank > 0) {
			MPI_Recv(&escapeFlag, 1, MPI_INT, MASTER_PROCESS, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}
		// Check stop condition on master
		else {
			/// This sum will later be parallelized
			T norm_diff_2 = 0;
			for (int i = 0; i < rowsHeldByRank * N; ++i) norm_diff_2 += sqr(x_ptr[i] - x0_ptr[i]);
			escapeFlag = (sqrt(norm_diff_2) < epsilon);

			RANKPRINT << "error = " << norm_diff_2 << std::endl; /// TEMP

			for (int rank = 1; rank < MPI_size; ++rank)
				MPI_Send(&escapeFlag, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
		}

	} while (!escapeFlag);

	// Return vector from last iteration
	if (x.get() == x_ptr) return x;
	else return x0;
}