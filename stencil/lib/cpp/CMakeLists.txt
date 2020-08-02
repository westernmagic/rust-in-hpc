string (TOLOWER "${CMAKE_CXX_COMPILER_ID}" COMPILER_ID)
if ("${COMPILER_ID}" STREQUAL "clang")
	set (COMPILER_ID "cray")
endif ()

add_library (
	stencil_cpp
		v0_base.cpp
		v1_inline.cpp
)
target_compile_definitions (
	stencil_cpp PRIVATE
		diffuse_v0_base=diffuse_${COMPILER_ID}_cpp_v0_base
		diffuse_v1_inline=diffuse_${COMPILER_ID}_cpp_v1_base
)
set_property (
	TARGET stencil_cpp
	PROPERTY OUTPUT_NAME stencil_${COMPILER_ID}_cpp
)
install (TARGETS stencil_cpp)
