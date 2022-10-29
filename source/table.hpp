#pragma once

#include <iostream>
#include <iomanip>
#include <string>


constexpr std::streamsize COL_SIZE_1 = 17;

template<typename Object>
void table_add_1(const Object &obj) {
	outstream
		<< " | "
		<< std::setw(COL_SIZE_1) << obj
		<< " | ";
}


constexpr std::streamsize COL_SIZE_2 = 12;

template<typename Object>
void table_add_2(const Object &obj) {
	outstream
		<< std::setw(COL_SIZE_2) << obj
		<< " | ";
}


constexpr std::streamsize COL_SIZE_3 = 12;

template<typename Object>
void table_add_3(const Object &obj) {
	outstream
		<< std::setw(COL_SIZE_3) << obj
		<< " | ";
}


constexpr std::streamsize COL_SIZE_4 = 12;

template<typename Object>
void table_add_4(const Object &obj) {
	outstream
		<< std::setw(COL_SIZE_4) << obj
		<< " | ";
}

constexpr std::streamsize COL_SIZE_5 = 12;

template<typename Object>
void table_add_5(const Object &obj) {
	outstream
		<< std::setw(COL_SIZE_5) << obj
		<< " |\n";
}


void table_hline() {
	std::string str1("");
	std::string str2("");
	std::string str3("");
	std::string str4("");
	std::string str5("");
	str1.insert(0, 1 + COL_SIZE_1 + 1, '-');
	str2.insert(0, 1 + COL_SIZE_2 + 1, '-');
	str3.insert(0, 1 + COL_SIZE_3 + 1, '-');
	str4.insert(0, 1 + COL_SIZE_4 + 1, '-');
	str5.insert(0, 1 + COL_SIZE_5 + 1, '-');

	outstream << " |" << str1 << "|" << str2 << "|" << str3 << "|" << str4 << "|" << str5 << "|\n";
}