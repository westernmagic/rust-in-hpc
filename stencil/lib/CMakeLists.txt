cmake_minimum_required (VERSION 3.18)
project (stencil_lib LANGUAGES Fortran CXX CUDA)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message (FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif ()

if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE "RelWithDebInfo")
endif ()

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include (Flags)

set (CMAKE_Fortran_FORMAT FREE)
set (BUILD_SHARED_LIBS TRUE)

set (CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

set (CMAKE_SKIP_BUILD_RPATH FALSE)
set (CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

set (CMAKE_CUDA_STANDARD 17)
set (CMAKE_CUDA_ARCHITECTURES 60)

set (CMAKE_CUDA_SEPARABLE_COMPILATION FALSE)
set (CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS FALSE)

find_package (OpenMP REQUIRED COMPONENTS CXX Fortran)

add_subdirectory (cpp)
add_subdirectory (f)
