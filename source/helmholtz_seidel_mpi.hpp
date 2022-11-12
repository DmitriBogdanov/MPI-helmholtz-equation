#pragma once

#include "communication.hpp"

// ### Helholtz equation ###
//
// >>> For full method description and notes check serial version in "helmholt_seidel_serial.hpp"
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
// Each rank computes '||x - x_0||' on its rows and sends resulting sums to master, which
// computes total '||x - x_0||' and checks stop condition '||x - x_0|| < epsilon', then
// master sends stop condition state to all other ranks so they exit the loop if needed
//
// NOTE: 'MPI_comminication_type' parameter selects one of following communication methods:
//       1) MPI_Send() + MPI_Reacv()
//       2) MPI_Sendrecv()
//       3) MPI_Isend() + MPI_Irecv()
//
inline UniquePtrArray helholtz_seidel_mpi(T k, std::function<T(T, T)> f, T L, size_t N, T epsilon,
	std::function<T(T)> boundary_left, std::function<T(T)> boundary_right,
	std::function<T(T)> boundary_bot, std::function<T(T)> boundary_top,
	int MPI_rank, int MPI_size, int MPI_comminication_type
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

	// Tabulate boundaries
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

	// Variables related to stop condition
	T propagated_norm_sum = 0;
	T local_norm_sum = 0;
	int escape_flag = 0; // MPI doesn't know bools exist

	// Seidel method
	do {
		// x0 = x
		std::swap(x_ptr, x0_ptr);
		for (int i = 1; i < rows_held_by_rank - 1; ++i) {
			for (int j = 1 + i % 2; j < N - 1; j += 2) {
				const int indexIJ = i * N + j;

				const T gridX = j * h;
				const T gridY = (MPI_rank * rows_processed_by_rank + i) * h; 

				x_ptr[indexIJ] = inverseAlpha * (
					x0_ptr[indexIJ - N] + // up
					x0_ptr[indexIJ + N] + // down
					x0_ptr[indexIJ - 1] + // left
					x0_ptr[indexIJ + 1] + // right
					beta * f(gridX, gridY)
					);
			}
		}

		// Syncronize state
		switch (MPI_comminication_type) {
		case 1:
			MPI_sync_rows_1(x_ptr, MPI_rank, MPI_size, rows_held_by_rank, N);
			break;
		case 2:
			MPI_sync_rows_2(x_ptr, MPI_rank, MPI_size, rows_held_by_rank, N);
			break;
		case 3:
			MPI_sync_rows_3(x_ptr, MPI_rank, MPI_size, rows_held_by_rank, N);
			MPI_Barrier(MPI_COMM_WORLD);
			break;
		default:
			exit_with_error("Unknown 'MPI_comminication_type' in Seidel method");
		}

		for (int i = 1; i < rows_held_by_rank - 1; ++i) {
			for (int j = 1 + (i + 1) % 2; j < N - 1; j += 2) {
				const int indexIJ = i * N + j;

				const T gridX = j * h;
				const T gridY = (MPI_rank * rows_processed_by_rank + i) * h; 

				x_ptr[indexIJ] = inverseAlpha * (
					x_ptr[indexIJ - N] + // up
					x_ptr[indexIJ + N] + // down
					x_ptr[indexIJ - 1] + // left
					x_ptr[indexIJ + 1] + // right
					beta * f(gridX, gridY)
					);
			}
		}

		// Syncronize state
		switch (MPI_comminication_type) {
		case 1:
			MPI_sync_rows_1(x_ptr, MPI_rank, MPI_size, rows_held_by_rank, N);
			break;
		case 2:
			MPI_sync_rows_2(x_ptr, MPI_rank, MPI_size, rows_held_by_rank, N);
			break;
		case 3:
			MPI_sync_rows_3(x_ptr, MPI_rank, MPI_size, rows_held_by_rank, N);
			break;
		default:
			exit_with_error("Unknown 'MPI_comminication_type' in Seidel method");
		}

		// Compute partial sum of ||x - x0||
		local_norm_sum = 0;
		for (int i = N; i < (rows_held_by_rank - 1) * N; ++i) local_norm_sum += sqr(x_ptr[i] - x0_ptr[i]);

		// Propagate partial sums to master, adding 'norm_diff_sum' to the received sum at each rank.
		//    {MPI_size - 1} -> {MPI_size - 2} -> ... -> {1} -> {0}
		// As a result we get total sum of ||x - x0|| without having to 'collect' the entire thing at one rank.
		//
		// NOTE: Other implementations are possible, but so far this one is the fastest
		//
		propagated_norm_sum = 0;
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


/// 'Lighweigh' stop condition without parallelization
        //// Get stop condition state from master and break out if needed
		//if (MPI_rank > 0) {
		//	MPI_Recv(&escapeFlag, 1, MPI_INT, MASTER_PROCESS, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
		//}
		//// Check stop condition on master
		//else {
		//	/// This sum will later be parallelized
		//	T norm_diff_2 = 0;
		//	for (int i = 0; i < rowsHeldByRank * N; ++i) norm_diff_2 += sqr(x_ptr[i] - x0_ptr[i]);
		//	escapeFlag = (sqrt(norm_diff_2) < epsilon);

		//	for (int rank = 1; rank < MPI_size; ++rank)
		//		MPI_Send(&escapeFlag, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);
		//}