cmake_minimum_required (VERSION 3.17)
project (stencil_lib LANGUAGES Fortran CXX)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message (FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif ()

if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE "RelWithDebInfo")
endif ()

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include (Flags)

set (CMAKE_Fortran_FORMAT FREE)
set (CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set (BUILD_SHARED_LIBS TRUE)

find_package (OpenMP REQUIRED COMPONENTS CXX Fortran)

add_subdirectory (cpp)
add_subdirectory (f)
