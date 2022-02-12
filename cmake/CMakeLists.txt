cmake_minimum_required (VERSION 3.17)
set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

project (RustCMake LANGUAGES Rust)

add_subdirectory (src)
