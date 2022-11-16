#pragma once

#include "utils.hpp"

// ### Helholtz equation ###
//
// >>> For full method description and notes check serial version in "helmholtz_jacobi_serial.hpp"
//
// ### MPI parallelization ###
//
// >>> For parallelization scheme check regular MPI version in "helmholtz_jacobi_mpi.hpp"
//
// ### Async communication ###
//
// Also known as non-blocking Send(), Recv()
// 
// Key idea: Compute first and last rows on each rank and start up async communication
//           while computing middle rows in the meantime
//
// NOTE: Due to the trick of swapping pointers 'x' and 'x0' to avoid copying data
//       MPI_..._init() has to be set up for both 'x' and 'x0'
//
inline UniquePtrArray helholtz_jacobi_mpi_async(T k, std::function<T(T, T)> f, T L, size_t N, T epsilon,
	std::function<T(T)> boundary_left, std::function<T(T)> boundary_right,
	std::function<T(T)> boundary_bot, std::function<T(T)> boundary_top,
	int MPI_rank, int MPI_size
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

	// -------------------------------------
	// --- Setup communication framework ---
	// -------------------------------------

	const auto x_rowTop = x_ptr;
	const auto x_rowBot = x_ptr + (rows_held_by_rank - 1) * N;
	const auto x_rowUpper = x_rowTop + N;
	const auto x_rowLower = x_rowBot - N;

	const auto x0_rowTop = x0_ptr;
	const auto x0_rowBot = x0_ptr + (rows_held_by_rank - 1) * N;
	const auto x0_rowUpper = x0_rowTop + N;
	const auto x0_rowLower = x0_rowBot - N;

	// 'Cycle' the ranks
	const bool isFirst = (MPI_rank == 0);
	const bool isLast = (MPI_rank == MPI_size - 1);

	const int prev_rank = isFirst ? MPI_size - 1 : MPI_rank - 1; // ternary can be optimized 
	const int next_rank = isLast ? 0 : MPI_rank + 1; // to branchless bitselect

	// Downwards wave
	const int recv_up_size = isFirst ? 0 : N;
	const int send_down_size = isLast ? 0 : N;

	// 1st set of inits
	MPI_Request request_downwards_send_x;
	MPI_Request request_downwards_recv_x;
	MPI_Send_init(x_rowLower, send_down_size, MPI_T, next_rank, 0, MPI_COMM_WORLD, &request_downwards_send_x);
	MPI_Recv_init(x_rowTop, recv_up_size, MPI_T, prev_rank, 0, MPI_COMM_WORLD, &request_downwards_recv_x);
	// 2nd set of inits
	MPI_Request request_downwards_send_x0;
	MPI_Request request_downwards_recv_x0;
	MPI_Send_init(x0_rowLower, send_down_size, MPI_T, next_rank, 0, MPI_COMM_WORLD, &request_downwards_send_x0);
	MPI_Recv_init(x0_rowTop, recv_up_size, MPI_T, prev_rank, 0, MPI_COMM_WORLD, &request_downwards_recv_x0);

	// Upwards wave
	const int recv_down_size = isLast ? 0 : N;
	const int send_up_size = isFirst ? 0 : N;

	// 1st set of inits
	MPI_Request request_upwards_send_x;
	MPI_Request request_upwards_recv_x;
	MPI_Send_init(x_rowUpper, send_up_size, MPI_T, prev_rank, 0, MPI_COMM_WORLD, &request_upwards_send_x);
	MPI_Recv_init(x_rowBot, recv_down_size, MPI_T, next_rank, 0, MPI_COMM_WORLD, &request_upwards_recv_x);
	// 2nd set of inits
	MPI_Request request_upwards_send_x0;
	MPI_Request request_upwards_recv_x0;
	MPI_Send_init(x0_rowUpper, send_up_size, MPI_T, prev_rank, 0, MPI_COMM_WORLD, &request_upwards_send_x0);
	MPI_Recv_init(x0_rowBot, recv_down_size, MPI_T, next_rank, 0, MPI_COMM_WORLD, &request_upwards_recv_x0);

	// Jacobi method
	bool ptrs_are_swapped = false; // used to track which set MPI inits has to be used
	do {
		// x0 = x
		std::swap(x_ptr, x0_ptr);
		ptrs_are_swapped = !ptrs_are_swapped;

		// ----------------------------
		// --- Internal Jacobi loop ---
		// ----------------------------

		// Compute first and last rows
		for (int i = 1; i < rows_held_by_rank - 1; i += rows_processed_by_rank - 1) {
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

		// Start downwards+upwards wave
		// (due to using async separate downwards/upwards 'waves' don't really matter)
		if (ptrs_are_swapped) {
			MPI_Start(&request_downwards_recv_x0);
			MPI_Start(&request_downwards_send_x0);

			MPI_Start(&request_upwards_recv_x0);
			MPI_Start(&request_upwards_send_x0);
		}
		else {
			MPI_Start(&request_downwards_recv_x);
			MPI_Start(&request_downwards_send_x);

			MPI_Start(&request_upwards_recv_x);
			MPI_Start(&request_upwards_send_x);
		}

		// Compute middle rows
		for (int i = 2; i < rows_held_by_rank - 2; ++i) {
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

		// Finish downwards+upwards wave
		if (ptrs_are_swapped) {
			MPI_Wait(&request_downwards_recv_x0, MPI_STATUSES_IGNORE);
			MPI_Wait(&request_downwards_send_x0, MPI_STATUSES_IGNORE);

			MPI_Wait(&request_upwards_recv_x0, MPI_STATUSES_IGNORE);
			MPI_Wait(&request_upwards_send_x0, MPI_STATUSES_IGNORE);
		}
		else {
			MPI_Wait(&request_downwards_recv_x, MPI_STATUSES_IGNORE);
			MPI_Wait(&request_downwards_send_x, MPI_STATUSES_IGNORE);

			MPI_Wait(&request_upwards_recv_x, MPI_STATUSES_IGNORE);
			MPI_Wait(&request_upwards_send_x, MPI_STATUSES_IGNORE);
		}

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