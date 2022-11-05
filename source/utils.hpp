#pragma once

#include <memory>
#include <cmath>
#include <iostream>
#include <fstream> // test

#pragma warning (disable: 26451) /// Fix for VS2019 bug displaying false warnings on integer multiplication


using T = double;
inline static auto &outstream = std::cout;

// Math
constexpr T PI = 3.14159265358979323846;

template<typename Type>
constexpr Type sqr(Type value) { return value * value; } // screw you C++, I want my sqr()


// 'Raw' array
using UniquePtrArray = std::unique_ptr<T[]>;

UniquePtrArray make_raw_array(size_t size) {
	return UniquePtrArray(new T[size]);
}

UniquePtrArray make_raw_array(size_t size, T defaultValue) {
	UniquePtrArray arr(new T[size]);
	for (int k = 0; k < size; ++k) arr[k] = defaultValue;
	return arr;
}

void print_array(T* const arr, size_t size) {
	outstream << "{ ";
	for (size_t k = 0; k < size - 1; ++k) outstream << arr[k] << ", ";
	outstream << arr[size - 1] << " }" << std::endl;
}


// Utility
inline void exit_with_error(const std::string &msg) {
	outstream << "ERROR: " << msg << "\n";
	exit(1);
}