cmake_minimum_required (VERSION 3.10)
project (fortran2rust LANGUAGES Fortran)

set (CMAKE_Fortran_FORMAT FREE)

list (APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake-cargo/cmake")
set (CARGO_DEV_MODE ON CACHE INTERNAL "")

include (Cargo)
add_crate (Cargo.toml)
add_subdirectory (examples)
