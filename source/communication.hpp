#pragma once
#include <functional>

#include <mpi.h>
#include "utils.hpp"
#define MPI_T MPI_DOUBLE
#define MASTER_PROCESS 0


// MPI Communication types
inline void MPI_sync_rows_1(T *x_ptr, int MPI_rank, int MPI_size, int rowsHeldByProcess, int N) {
	const auto rowTop = x_ptr;
	const auto rowBot = x_ptr + (rowsHeldByProcess - 1) * N;
	const auto rowUpper = rowTop + N;
	const auto rowLower = rowBot - N;

	// 'Cycle' the ranks
	const bool isFirst = (MPI_rank == 0);
	const bool isLast  = (MPI_rank == MPI_size - 1);

	const int prev_rank = isFirst ? MPI_size - 1 : MPI_rank - 1; // ternary can be optimized 
	const int next_rank = isLast  ? 0            : MPI_rank + 1; // to branchless bitselect

	// Downwards wave
	const int recv_up_size   = isFirst ? 0 : N;
	const int send_down_size =  isLast ? 0 : N;

	MPI_Send(rowLower, send_down_size, MPI_T, next_rank, 0, MPI_COMM_WORLD);
	MPI_Recv(  rowTop,   recv_up_size, MPI_T, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
	
	// Upwards wave
	const int recv_down_size =  isLast ? 0 : N;
	const int send_up_size   = isFirst ? 0 : N;

	MPI_Send(rowUpper, send_up_size, MPI_T, prev_rank, 0, MPI_COMM_WORLD);
	MPI_Recv(  rowBot, recv_down_size, MPI_T, next_rank, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
}

inline void MPI_sync_rows_2(T *x_ptr, int MPI_rank, int MPI_size, int rowsHeldByProcess, int N) {
	const auto rowTop = x_ptr;
	const auto rowBot = x_ptr + (rowsHeldByProcess - 1) * N;
	const auto rowUpper = rowTop + N;
	const auto rowLower = rowBot - N;

	// 'Cycle' the ranks
	const bool isFirst = (MPI_rank == 0);
	const bool isLast  = (MPI_rank == MPI_size - 1);

	const int prev_rank = isFirst ? MPI_size - 1 : MPI_rank - 1; // ternary can be optimized 
	const int next_rank = isLast  ? 0            : MPI_rank + 1; // to branchless bitselect

	// Downwards wave
	const int recv_up_size   = isFirst ? 0 : N;
	const int send_down_size = isLast  ? 0 : N;

	MPI_Sendrecv(
		rowLower, send_down_size, MPI_T, next_rank, 0,
		  rowTop,   recv_up_size, MPI_T, prev_rank, 0,
		MPI_COMM_WORLD, MPI_STATUSES_IGNORE
	);

	// Upwards wave
	const int recv_down_size = isLast  ? 0 : N;
	const int send_up_size   = isFirst ? 0 : N;

	MPI_Sendrecv(
		rowUpper,   send_up_size, MPI_T, prev_rank, 0,
		  rowBot, recv_down_size, MPI_T, next_rank, 0,
		MPI_COMM_WORLD, MPI_STATUSES_IGNORE
	);
}

inline void MPI_sync_rows_3(T *x_ptr, int MPI_rank, int MPI_size, int rowsHeldByProcess, int N) {
	const auto rowTop = x_ptr;
	const auto rowBot = x_ptr + (rowsHeldByProcess - 1) * N;
	const auto rowUpper = rowTop + N;
	const auto rowLower = rowBot - N;

	MPI_Request REQUEST_DISCARD; // here we put requests and completion bools that are to be discarded
	int COMPLETION_DISCARD = 0;  // (MPI does not have in-built discard option for these)

	MPI_Request request_send_below;
	MPI_Request request_send_above;

	// MPI_Recv() 1  upper row from the process {MPI_rank - 1}
	if (MPI_rank > 0) MPI_Irecv(
		rowTop, N, MPI_T, MPI_rank - 1, 0, MPI_COMM_WORLD, &REQUEST_DISCARD
	);

	// MPI_Send() 1 bottom row to the process {MPI_rank + 1}
	if (MPI_rank < MPI_size - 1) MPI_Isend(
		rowLower, N, MPI_T, MPI_rank + 1, 0, MPI_COMM_WORLD, &request_send_below
	);

	// MPI_Recv() 1 bottom row from the process{ MPI_rank + 1 }
	if (MPI_rank < MPI_size - 1) MPI_Irecv(
		rowBot, N, MPI_T, MPI_rank + 1, 0, MPI_COMM_WORLD, &REQUEST_DISCARD
	);

	// MPI_Send() 1  upper row to the process {MPI_rank - 1}
	if (MPI_rank > 0) MPI_Isend(
		rowUpper, N, MPI_T, MPI_rank - 1, 0, MPI_COMM_WORLD, &request_send_above
	);

	if (MPI_rank < MPI_size - 1)
	{
		MPI_Test(&request_send_below, &COMPLETION_DISCARD, MPI_STATUSES_IGNORE);
		//MPI_Waitall(MPI_size, &request_send_below, MPI_STATUSES_IGNORE);
		///MPI_Wait(&request_send_below, MPI_STATUSES_IGNORE);
	};
	if (MPI_rank > 0)
	{
		MPI_Test(&request_send_above, &COMPLETION_DISCARD, MPI_STATUSES_IGNORE);
		//MPI_Waitall(MPI_size, &request_send_above, MPI_STATUSES_IGNORE);
		///MPI_Wait(&request_send_above, MPI_STATUSES_IGNORE);
	};
}