#include <iostream>
#include <iomanip>

#include "static_timer.hpp"
#include "helmholtz_jacobi_mpi.hpp"
#include "table.hpp"

#include <mpi.h>

#define SEQUENTIAL_MODE 1


// # Config #
// Area size
const T L = 1;

// Grid size
const size_t N = 6002;
const size_t pointCount = sqr(N);

// Wave number
const T c1 = 10;
const T h = L / (N - 1);

const T k = sqrt(c1) / h;

// Precision
const T precision = 1e-6;

// Right part
T right_part(T x, T y) { return(2 * sin(PI * y) + k * k * (1 - x) * x * sin(PI * y) + PI * PI * (1 - x) * x * sin(PI * y)); }

// Boundaries
T zero_boundary(T value) { return 0; }

// Precise solution (for analythical purposes)
T analythical_solution(T x, T y) { return (1 - x) * x * sin(PI * y); }

UniquePtrArray get_exact_solution() {
	auto sol_exact = make_raw_array(pointCount);

	T h = T(1) / (N - 1);

	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
			sol_exact[i * N + j] = analythical_solution(j * h, i * h);

	return sol_exact;
}

//T get_relative_error_L2(T* const sol_numeric, T* const sol_exact) {
//	T L2_norm_of_difference(0);
//	T L2_norm_of_exact_sol(0);
//
//	for (size_t i = 0; i < pointCount; ++i) {
//		L2_norm_of_difference += sqr(sol_numeric[i] - sol_exact[i]);
//		L2_norm_of_exact_sol += sqr(sol_exact[i]);
//	}
//
//	return sqrt(L2_norm_of_difference / L2_norm_of_exact_sol);
//}

T get_relative_error_L2(T* const sol_numeric, int MPI_rank, int MPI_size) {
	const int internalN = N - 2;

	const int rowsProcessedByRank = internalN / MPI_size;
	const int rowsHeldByRank = rowsProcessedByRank + 2;

	T norms[2] = { 0 , 0 };
		// [1] -> norm ||x - x_numeric||
		// [2] -> norm ||x||
		// Held as an array for convenient MPI sending

	for (int i = 1; i < rowsHeldByRank - 1; ++i) {
		for (int j = 0; j < N; ++j) {
			const int indexIJ = i * N + j;

			const T gridX = j * h;
			const T gridY = (MPI_rank * rowsProcessedByRank + i) * h; // gotta account for current block position

			norms[0] += sqr(sol_numeric[indexIJ] - analythical_solution(gridX, gridY));
			norms[1] += sqr(analythical_solution(gridX, gridY));
		}
	}

	RANKPRINT << "norms = {" << norms[0] << ", " << norms[1] << "}" << std::endl;

	// Collect all ||x - x_numeric|| and ||x|| norms on master and add them together
	if (MPI_rank == 0) {
		T temp[2];

		for (int rank = 1; rank < MPI_size; ++rank) {
			RANKPRINT << "awaiting norm from " << rank << std::endl;

			MPI_Recv(temp, 2, MPI_T, rank, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

			RANKPRINT << "received norm from " << rank << std::endl;

			norms[0] += temp[0];
			norms[1] += temp[1];
		}

		RANKPRINT << "coolected norms = {" << norms[0] << ", " << norms[1] << "}" << std::endl;
	}
	else {
		MPI_Send(norms, 2, MPI_T, MASTER_PROCESS, 0, MPI_COMM_WORLD);

		RANKPRINT << "sent norms" << std::endl;
	}

	// Relative error  ||x - x_numeric|| / ||x||
	return sqrt(norms[0] / norms[1]);
}

void print_sol(T* const sol) {
	for (size_t i = 0; i < N; ++i) {
		outstream << "[";
		for (size_t j = 0; j < N; ++j)
			outstream << std::setw(15) << sol[i * N + j];
		outstream << "]\n";
	}
}


int main(int argc, char** argv) {
	// Initialize MPI environment
	MPI_Init(&argc, &argv);

	// Get rank of the process
	int MPI_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);

	int MPI_size;
	MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);

	outstream << "Rank " << MPI_rank << "/" << MPI_size << " initialized\n";

	// Sync
	{
		auto solution = helholtz_jacobi(
			k, right_part, L, N, precision,
			zero_boundary, zero_boundary, zero_boundary, zero_boundary,
			MPI_rank, MPI_size, 1
		);

		auto error = get_relative_error_L2(solution.get(), MPI_rank, MPI_size);

		if (MPI_rank == 0) {
			std::cout << "Relative error = " << error << std::endl;
		}
	}

	/*if (MPI_rank > 0) {
		_sleep(1'000);
		int temp = 0;

		MPI_Recv(&temp, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &MPI_status);

		outstream << "Rank " << MPI_rank << "/" << MPI_size << " synced" << std::endl;
	}
	else {
		_sleep(1'000);
		for (int rank = 1; rank < MPI_size; ++rank)
			MPI_Send(&TRUE, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);

		outstream << "Rank " << MPI_rank << "/" << MPI_size << " synced" << std::endl;
		_sleep(2'000);
	}*/

	/// TEMPs
	//_sleep(1000);
	
	// Finalize MPI environment
	outstream << "Rank " << MPI_rank << " finalized\n";
	MPI_Finalize();
}

