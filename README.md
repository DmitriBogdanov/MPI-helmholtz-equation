# MPI Helmholtz Equation

Contains serial and parallel (MPI) implementations of following algorithms:

* Helholtz equation solver (internal iterations using Jacobi method)
* Helholtz equation solver (internal iterations using Seidel method with black-red ordering)

Note that present implementations are intended for academic purposes, as such they are not meant to be used in any sort of high-performance production code.

## Compilation

* Recommended compiler: Intel C++ Compiler
* Requires C++17 support
* Requires MPI implementation

## Usage

Helmholtz equation:
-u_xx - u_yy + k^2 u(x, y) = f(x, y)
where
k - wave_number
f - right_part
defined on a 2D region [0, L]x[0, L] with 4 fist-type boundary conditions

Adjust grid size N, area size L, wave number k, precision and boundary conditions in "main.cpp" to configure testing parameters. Parallel implementations assume (N - 2) to be a multiple of MPI_Comm_size().

## Version history

* 00.01
    * Implemented parallel Jacobi method with MPI (stop condition is serial for now)
    * Implemented parallel calculation of relative error

## License

This project is licensed under the MIT License - see the LICENSE.md file for details