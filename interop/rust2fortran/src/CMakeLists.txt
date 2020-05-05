cmake_minimum_required (VERSION 3.10)
project (rust2fortran LANGUAGES Fortran)
set (CMAKE_Fortran_FORMAT FREE)

add_library (zaxpy zaxpy.f)
install (TARGETS zaxpy)
