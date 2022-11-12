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