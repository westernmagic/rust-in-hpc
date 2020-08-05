string (TOLOWER "${CMAKE_CXX_COMPILER_ID}" COMPILER_ID)
if ("${COMPILER_ID}" STREQUAL "clang")
	set (COMPILER_ID "cray")
endif ()

add_library (
	stencil_cpp
		laplap_seq.cpp
		inline_seq.cpp
		laplap_openmp.cpp
		inline_openmp.cpp
		laplap_openacc.cpp
		inline_openacc.cpp
		laplap_openmp_target.cpp
		inline_openmp_target.cpp
)
target_compile_definitions (
	stencil_cpp PRIVATE
		diffuse_laplap_seq=diffuse_${COMPILER_ID}_cpp_laplap_seq
		diffuse_inline_seq=diffuse_${COMPILER_ID}_cpp_inline_seq
		diffuse_laplap_openmp=diffuse_${COMPILER_ID}_cpp_laplap_openmp
		diffuse_inline_openmp=diffuse_${COMPILER_ID}_cpp_inline_openmp
		diffuse_laplap_openacc=diffuse_${COMPILER_ID}_cpp_laplap_openacc
		diffuse_inline_openacc=diffuse_${COMPILER_ID}_cpp_inline_openacc
		diffuse_laplap_openmp_target=diffuse_${COMPILER_ID}_cpp_laplap_openmp_target
		diffuse_inline_openmp_target=diffuse_${COMPILER_ID}_cpp_inline_openmp_target
)
target_link_libraries (
	stencil_cpp
		OpenMP::CXX
		OpenACC::CXX
)
set_property (
	TARGET stencil_cpp
	PROPERTY OUTPUT_NAME stencil_${COMPILER_ID}_cpp
)
install (TARGETS stencil_cpp)
