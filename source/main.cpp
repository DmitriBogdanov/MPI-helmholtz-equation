#include <iostream>
#include <iomanip>

#include "helmholtz_jacobi_serial.hpp"
#include "helmholtz_jacobi_mpi.hpp"
#include "helmholtz_jacobi_mpi_async.hpp"

#include "helmholtz_seidel_serial.hpp"
#include "helmholtz_seidel_mpi.hpp"
#include "helmholtz_seidel_mpi_async.hpp"

#include "static_timer.hpp"
#include "table.hpp"


// # Config #
// Area size
const T L = 1;

// Grid size
const size_t N = 10002;
const size_t internalN = N - 2;

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


// Solution verification
T get_relative_error_serial(T* const sol_numeric) {
	T L2_norm_of_difference(0);
	T L2_norm_of_exact_sol(0);

	for (int i = 1; i < N - 1; ++i) {
		for (int j = 0; j < N; ++j) {
			const int indexIJ = i * N + j;

			const T precise_value = analythical_solution(j * h, i * h);

			L2_norm_of_difference += sqr(sol_numeric[indexIJ] - precise_value);
			L2_norm_of_exact_sol += sqr(precise_value);
		}
	}

	return sqrt(L2_norm_of_difference / L2_norm_of_exact_sol);
}

T get_relative_error_mpi(T* const sol_numeric, int MPI_rank, int MPI_size) {
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

			const T precise_value = analythical_solution(gridX, gridY);

			norms[0] += sqr(sol_numeric[indexIJ] - precise_value);
			norms[1] += sqr(precise_value);
		}
	}

	// Collect all ||x - x_numeric|| and ||x|| norms on master and add them together
	if (MPI_rank == 0) {
		T temp[2];

		for (int rank = 1; rank < MPI_size; ++rank) {
			MPI_Recv(temp, 2, MPI_T, rank, 0, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

			norms[0] += temp[0];
			norms[1] += temp[1];
		}
	}
	else {
		MPI_Send(norms, 2, MPI_T, MASTER_PROCESS, 0, MPI_COMM_WORLD);
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

	outstream << "Rank " << MPI_rank << "/" << MPI_size << " initialized\n" << std::flush;

	// Draw a table
	MPI_Barrier(MPI_COMM_WORLD);
	if (MPI_rank == 0) {
		std::cout
			<< "\n"
			<< "N = " << N << "\n"
			<< "k^2 h^2 = " << c1 << "\n"
			<< "precision = " << precision << "\n\n"
			<< std::flush;
	}

	if (internalN % MPI_size != 0) exit_with_error("intenalN is not divisible by MPI_size");

	if (MPI_rank == 0) {
		table_add_1("Method");
		table_add_2("Time (sec)");
		table_add_3("Rel. Error");
		table_add_4("Speedup");
		table_hline();
	}

	double jacobiSerialTime   = -1;
	double jacobiParallelTime = -1;
	double seidelSerialTime   = -1;
	double seidelParallelTime = -1;

	// 1) Serial Jacobi (MPI communication type 1)
	if (MPI_rank == 0) {
		// 1. Method
		table_add_1("Jacobi");

		// 2. Time
		StaticTimer::start();
		auto solution = helholtz_jacobi_serial(
			k, right_part, L, N, precision,
			zero_boundary, zero_boundary, zero_boundary, zero_boundary
		);
		jacobiSerialTime = StaticTimer::end();

		table_add_2(jacobiSerialTime);

		// 3. Err
		table_add_3(get_relative_error_serial(solution.get()));

		// 4. Speedup
		table_add_4(jacobiSerialTime / jacobiSerialTime);
	}

	// 2) Parallel Jacobi
	MPI_Barrier(MPI_COMM_WORLD);
	for (int MPI_communication_type = 1; MPI_communication_type <= 3; ++MPI_communication_type) {
		if (MPI_rank == 0) {
			// 1. Method
			table_add_1("MPI Jacobi (type " + std::to_string(MPI_communication_type) + ")");

			// 2. Time
			StaticTimer::start();
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		auto solution =
			(MPI_communication_type < 3)
			? helholtz_jacobi_mpi(
				k, right_part, L, N, precision,
				zero_boundary, zero_boundary, zero_boundary, zero_boundary,
				MPI_rank, MPI_size, MPI_communication_type
			)
			: helholtz_jacobi_mpi_async(
				k, right_part, L, N, precision,
				zero_boundary, zero_boundary, zero_boundary, zero_boundary,
				MPI_rank, MPI_size
			);
			
		MPI_Barrier(MPI_COMM_WORLD);

		if (MPI_rank == 0) {
			jacobiParallelTime = StaticTimer::end();

			table_add_2(jacobiParallelTime);
		}

		auto error = get_relative_error_mpi(solution.get(), MPI_rank, MPI_size);

		if (MPI_rank == 0) {
			// 3. Err
			table_add_3(error);

			// 4. Speedup
			table_add_4(jacobiSerialTime / jacobiParallelTime);
		}	
	}

	// 3) Serial Seidel (MPI communication type 1)
	if (MPI_rank == 0) {
		// 1. Method
		table_add_1("Seidel");

		// 2. Time
		StaticTimer::start();
		auto solution = helholtz_seidel_serial(
			k, right_part, L, N, precision,
			zero_boundary, zero_boundary, zero_boundary, zero_boundary
		);
		seidelSerialTime = StaticTimer::end();

		table_add_2(seidelSerialTime);

		// 3. Err
		table_add_3(get_relative_error_serial(solution.get()));

		// 4. Speedup
		table_add_4(jacobiSerialTime / seidelSerialTime);
	}

	// 2) Parallel Seidel
	MPI_Barrier(MPI_COMM_WORLD);
	for (int MPI_communication_type = 1; MPI_communication_type <= 3; ++MPI_communication_type) {
		if (MPI_rank == 0) {
			// 1. Method
			table_add_1("MPI Seidel (type " + std::to_string(MPI_communication_type) + ")");

			// 2. Time
			StaticTimer::start();
		}

		MPI_Barrier(MPI_COMM_WORLD);
		auto solution = (MPI_communication_type < 3)
			? helholtz_seidel_mpi(
				k, right_part, L, N, precision,
				zero_boundary, zero_boundary, zero_boundary, zero_boundary,
				MPI_rank, MPI_size, MPI_communication_type
			)
			: helholtz_seidel_mpi_async(
				k, right_part, L, N, precision,
				zero_boundary, zero_boundary, zero_boundary, zero_boundary,
				MPI_rank, MPI_size
			);
		MPI_Barrier(MPI_COMM_WORLD);

		if (MPI_rank == 0) {
			seidelParallelTime = StaticTimer::end();

			table_add_2(seidelParallelTime);
		}

		auto error = get_relative_error_mpi(solution.get(), MPI_rank, MPI_size);

		if (MPI_rank == 0) {
			// 3. Err
			table_add_3(error);

			// 4. Speedup
			table_add_4(seidelSerialTime / seidelParallelTime);
		}
	}
	
	// Finalize MPI environment
	if (MPI_rank == 0) std::cout << '\n' << std::flush;
	MPI_Barrier(MPI_COMM_WORLD);
	outstream << "Rank " << MPI_rank << " finalized\n" << std::flush;

	MPI_Finalize();
}

