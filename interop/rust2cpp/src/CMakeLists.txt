cmake_minimum_required (VERSION 3.10)
project (rust2cpp LANGUAGES CXX)

add_library (zaxpy zaxpy.cpp)
install (TARGETS zaxpy)
