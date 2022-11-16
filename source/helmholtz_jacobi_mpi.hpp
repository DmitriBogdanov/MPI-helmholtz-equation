#pragma once

#include "utils.hpp"

// ### Helholtz equation ###
//
// >>> For full method description and notes check serial version in "helmholtz_jacobi_serial.hpp"
//
// ### MPI parallelization ###
//
// Example for
//   N = 10
//   => internalN = N - 2 = 8
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
// Each rank computes partial sum ||x - x_0||_L2 on its rows and sends resulting sums to master, which
// computes total '||x - x_0||' and checks stop condition '||x - x_0|| < epsilon', then
// master broadcasts stop condition state to all other ranks so they exit the loop if needed
//
// ### Row syncronisation scheme ###
// 
// Example for (MPI_size == 4):
//
//   send 0 bytes to {last}          recv 0 bytes from {last}
//   |                               |
//    <---  l 0 0 0 0 0 0 0 0 r  <---
//          l 0 0 0 0 0 0 0 0 r
//          ...................          RANK 0 / 4
//          l 0 0 0 0 0 0 0 0 r  --->
//    --->  l 0 0 0 0 0 0 0 0 r      |
//   |                               |
//   |                               |
//   |      l 0 0 0 0 0 0 0 0 r  <---
//    <---  l 0 0 0 0 0 0 0 0 r
//          ...................          RANK 1 / 4
//          l 0 0 0 0 0 0 0 0 r  --->
//    --->  l 0 0 0 0 0 0 0 0 r      |
//   |                               |
//   |                               |
//   |      l 0 0 0 0 0 0 0 0 r  <---
//    <---  l 0 0 0 0 0 0 0 0 r
//          ...................          RANK 2 / 4
//          l 0 0 0 0 0 0 0 0 r
//    --->  l 0 0 0 0 0 0 0 0 r  --->
//   |                               |
//   |                               |
//   |      l 0 0 0 0 0 0 0 0 r  <---
//    <---  l 0 0 0 0 0 0 0 0 r
//          ...................          RANK 3 / 4
//          l 0 0 0 0 0 0 0 0 r
//    --->  l 0 0 0 0 0 0 0 0 r  <---
//   |                               |
//   recv 0 bytes from {first}       send 0 bytes to {first}
//
// NOTE: 'MPI_comminication_type' parameter selects one of following communication methods:
//       1) MPI_Send() + MPI_Reacv()
//       2) MPI_Sendrecv()
//
inline UniquePtrArray helholtz_jacobi_mpi(T k, std::function<T(T, T)> f, T L, size_t N, T epsilon,
	std::function<T(T)> boundary_left, std::function<T(T)> boundary_right,
	std::function<T(T)> boundary_bot, std::function<T(T)> boundary_top,
	int MPI_rank, int MPI_size, int MPI_comminication_type /* commtype 1 or 2 */
) {
	T h = T(1) / (N - 1);

	const T alpha = T(4) + sqr(h) * sqr(k);
	const T inverseAlpha = T(1) / alpha;
	const T beta = sqr(h);

	const int internalN = N - 2;

	const int rows_processed_by_rank = internalN / MPI_size;
	const int rows_held_by_rank = rows_processed_by_rank + 2;

	// Each process holds (internalN / MPI_size + 2) rows of 'X' and 'X0'
	auto x = make_raw_array(rows_held_by_rank * N, 0); // initial guess in zero-vector
	auto x0 = make_raw_array(rows_held_by_rank * N);

	auto x_ptr = x.get();
	auto x0_ptr = x0.get();

	// ---------------------------
	// --- Tabulate boundaries ---
	// ---------------------------

	// Top (held by process {0})
	if (MPI_rank == 0) for (int j = 1; j < N - 1; ++j) {
		x_ptr[0 * N + j] = boundary_top(j * h);
		x0_ptr[0 * N + j] = boundary_top(j * h);
	}
	// Bottom (held by process {MPI_size - 1})
	else if (MPI_rank == MPI_size - 1) for (int j = 1; j < N - 1; ++j) {
		x_ptr[(rows_held_by_rank - 1) * N + j] = boundary_bot(j * h);
		x0_ptr[(rows_held_by_rank - 1) * N + j] = boundary_bot(j * h);
	}
	// Left + Right (held by any other process)
	else for (int i = 0; i < rows_held_by_rank; ++i) {
		x_ptr[i * N + 0] = boundary_left(i * h);
		x0_ptr[i * N + 0] = boundary_left(i * h);

		x_ptr[i * N + N - 1] = boundary_right(i * h);
		x0_ptr[i * N + N - 1] = boundary_right(i * h);
	}

	int escape_flag = 0; // MPI doesn't know bools exist

	// ------------------------------------
	// --- Init communication framework ---
	// ------------------------------------

	// 'Cycle' the ranks
	const bool isFirst = (MPI_rank == 0);
	const bool isLast = (MPI_rank == MPI_size - 1);

	const int prev_rank = isFirst ? MPI_size - 1 : MPI_rank - 1; // ternary can be optimized 
	const int next_rank = isLast ? 0 : MPI_rank + 1; // to branchless bitselect

	// Downwards wave
	const int recv_up_size = isFirst ? 0 : N;
	const int send_down_size = isLast ? 0 : N;

	// Upwards wave
	const int recv_down_size = isLast ? 0 : N;
	const int send_up_size = isFirst ? 0 : N;

	// Jacobi method
	do {
		// x0 = x
		std::swap(x_ptr, x0_ptr);

		// ---------------------------------
		// --- Internal Jacobi iteration ---
		// ----------------------------------

		for (int i = 1; i < rows_held_by_rank - 1; ++i) {
			for (int j = 1; j < N - 1; ++j) {
				const int indexIJ = i * N + j;

				const T gridX = j * h;
				const T gridY = (MPI_rank * rows_processed_by_rank + i) * h; // gotta account for current block position

				x_ptr[indexIJ] = inverseAlpha * (
					x0_ptr[indexIJ - N] + // up
					x0_ptr[indexIJ + N] + // down
					x0_ptr[indexIJ - 1] + // left
					x0_ptr[indexIJ + 1] + // right
					beta * f(gridX, gridY)
					);
			}
		}

		// ------------------------
		// --- Syncronize state ---
		// ------------------------

		const auto rowTop = x_ptr;
		const auto rowBot = x_ptr + (rows_held_by_rank - 1) * N;
		const auto rowUpper = rowTop + N;
		const auto rowLower = rowBot - N;

		// Downwards wave
		if (MPI_comminication_type == 1) {
			MPI_Send(rowLower, send_down_size, MPI_T, next_rank, 0, MPI_COMM_WORLD);
			MPI_Recv(rowTop, recv_up_size, MPI_T, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}
		else MPI_Sendrecv(
			rowLower, send_down_size, MPI_T, next_rank, 0,
			rowTop, recv_up_size, MPI_T, prev_rank, 0,
			MPI_COMM_WORLD, MPI_STATUSES_IGNORE
		);

		// Upwards wave
		if (MPI_comminication_type == 1) {
			MPI_Send(rowUpper, send_up_size, MPI_T, prev_rank, 0, MPI_COMM_WORLD);
			MPI_Recv(rowBot, recv_down_size, MPI_T, next_rank, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		}
		else MPI_Sendrecv(
			rowUpper, send_up_size, MPI_T, prev_rank, 0,
			rowBot, recv_down_size, MPI_T, next_rank, 0,
			MPI_COMM_WORLD, MPI_STATUSES_IGNORE
		);
		
		// ----------------------
		// --- Stop Condition ---
		// ----------------------

		// Compute partial sum of ||x - x0|| on processed rows
		T local_norm_sum = 0;
		for (int i = N; i < (rows_held_by_rank - 1) * N; ++i) local_norm_sum += sqr(x_ptr[i] - x0_ptr[i]);

		// Propagate partial sums to master, adding 'norm_diff_sum' to the received sum at each rank.
		//    {MPI_size - 1} -> {MPI_size - 2} -> ... -> {1} -> {0}
		// As a result we get total sum of ||x - x0|| without having to 'collect' the entire thing at one rank.
		//
		// NOTE: Other implementations are possible, but so far this one is the fastest
		//
		T propagated_norm_sum = 0;
		if (MPI_rank < MPI_size - 1)
			MPI_Recv(&propagated_norm_sum, 1, MPI_T, MPI_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

		propagated_norm_sum += local_norm_sum;

		if (MPI_rank > 0)
			MPI_Send(&propagated_norm_sum, 1, MPI_T, MPI_rank - 1, 0, MPI_COMM_WORLD);
		else
			escape_flag = (sqrt(propagated_norm_sum) < epsilon);

		MPI_Bcast(&escape_flag, 1, MPI_INT, MASTER_PROCESS, MPI_COMM_WORLD);

	} while (!escape_flag);

	// Return vector from last iteration
	if (x.get() == x_ptr) return x;
	else return x0;
}