cmake_minimum_required (VERSION 3.10)
project (gemm_perf2 LANGUAGES CXX)

add_library (
	gemm_perf2
	mm00/mod.cpp
	mm01/mod.cpp
	mm02/mod.cpp
	mm03/mod.cpp
	mm04/mod.cpp
	mm05/mod.cpp
	mm06/mod.cpp
	mm07/mod.cpp
	mm08/mod.cpp
	mm09/mod.cpp
)

install (TARGETS gemm_perf2)
