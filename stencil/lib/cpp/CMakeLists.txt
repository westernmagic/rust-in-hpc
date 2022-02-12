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
)
target_compile_definitions (
	stencil_cpp PRIVATE
		diffuse_laplap_seq=diffuse_${COMPILER_ID}_cpp_laplap_seq
		diffuse_inline_seq=diffuse_${COMPILER_ID}_cpp_inline_seq
		diffuse_laplap_openmp=diffuse_${COMPILER_ID}_cpp_laplap_openmp
		diffuse_inline_openmp=diffuse_${COMPILER_ID}_cpp_inline_openmp
)
target_link_libraries (
	stencil_cpp
		OpenMP
)
if (NOT (
	"${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" OR
	"${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
))
	target_sources (
		stencil_cpp PRIVATE
			laplap_openmp_target.cpp
			inline_openmp_target.cpp
	)
	target_compile_definitions (
		stencil_cpp PRIVATE
			diffuse_laplap_openmp_target=diffuse_${COMPILER_ID}_cpp_laplap_openmp_target
			diffuse_inline_openmp_target=diffuse_${COMPILER_ID}_cpp_inline_openmp_target
	)
endif ()

if (NOT (
	"${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" OR
	"${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR
	"${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
))
	target_sources (
		stencil_cpp PRIVATE
			laplap_openacc.cpp
			inline_openacc.cpp
	)
	target_compile_definitions (
		stencil_cpp PRIVATE
			diffuse_laplap_openacc=diffuse_${COMPILER_ID}_cpp_laplap_openacc
			diffuse_inline_openacc=diffuse_${COMPILER_ID}_cpp_inline_openacc
	)
	target_link_libraries (
		stencil_cpp
			OpenACC
	)
endif ()

if (NOT ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI"))
	target_sources (
		stencil_cpp PRIVATE
			laplap_cuda.cu
			inline_cuda.cu
	)
	target_compile_definitions (
		stencil_cpp PRIVATE
			diffuse_laplap_cuda=diffuse_${COMPILER_ID}_cpp_laplap_cuda
			diffuse_inline_cuda=diffuse_${COMPILER_ID}_cpp_inline_cuda
	)
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set_source_files_properties (laplap_cuda.cu LANGUAGE CXX)
		set_source_files_properties (laplap_cuda.cu COMPILE_OPTIONS -x cuda)
		set_source_files_properties (inline_cuda.cu LANGUAGE CXX)
		set_source_files_properties (inline_cuda.cu COMPILE_OPTIONS -x cuda)
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
		set_source_files_properties (laplap_cuda.cu LANGUAGE CXX)
		set_source_files_properties (inline_cuda.cu LANGUAGE CXX)
		target_link_options (
			stencil_cpp PRIVATE
				-Mcuda=cc60,cuda10.1
		)
	endif ()
endif ()

set_property (
	TARGET stencil_cpp
	PROPERTY OUTPUT_NAME stencil_${COMPILER_ID}_cpp
)
install (TARGETS stencil_cpp)
