![C++](https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=cplusplus) <br>
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

Helmholtz equation:<br>
$$-u_{xx} - u_{yy} + k^2 u(x, y) = f(x, y),$$<br>
where<br>
$k$ - wave_number,<br>
$f$ - right_part,<br>
defined on a 2D region $[0, L]\times[0, L]$ with $4$ fist-type boundary conditions

Adjust grid size $N$, area size $L$, wave number $k$, precision and boundary conditions in "main.cpp" to configure testing parameters. Parallel implementations assume $(N - 2)$ to be a multiple of MPI_Comm_size().

## Version history

* 00.03
    * Added serial Jacobi implementation
    * Added multiple types of MPI communication, that can be toggled as a module inside the method
    * Added proper output in form of a table
    * Parallelized Jacobi stop condition
    * Finalized Jacobi implementation and got proper speedup from parallelization

* 00.02
    * Included full method description in comments

* 00.01
    * Implemented parallel Jacobi method with MPI (stop condition is serial for now)
    * Implemented parallel calculation of relative error

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
