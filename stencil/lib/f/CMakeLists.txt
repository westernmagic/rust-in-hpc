string (TOLOWER "${CMAKE_Fortran_COMPILER_ID}" COMPILER_ID)

add_library (
	stencil_f
		m_assert.f
		laplap_seq.f
		inline_seq.f
		laplap_openmp.f
		inline_openmp.f
)
target_compile_definitions (
	stencil_f PRIVATE
		diffuse_laplap_seq=diffuse_${COMPILER_ID}_f_laplap_seq
		diffuse_inline_seq=diffuse_${COMPILER_ID}_f_inline_seq
		diffuse_laplap_openmp=diffuse_${COMPILER_ID}_f_laplap_openmp
		diffuse_inline_openmp=diffuse_${COMPILER_ID}_f_inline_openmp
)
target_link_libraries (
	stencil_f
		OpenMP
)

if (NOT (
	"${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel" OR
	("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Cray" AND "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo") OR
	"${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU"
))
	target_sources (
		stencil_f PRIVATE
			laplap_openmp_target.f
			inline_openmp_target.f
			laplap_openacc.f
			inline_openacc.f
	)
	target_compile_definitions (
		stencil_f PRIVATE
			diffuse_laplap_openmp_target=diffuse_${COMPILER_ID}_f_laplap_openmp_target
			diffuse_inline_openmp_target=diffuse_${COMPILER_ID}_f_inline_openmp_target
			diffuse_laplap_openacc=diffuse_${COMPILER_ID}_f_laplap_openacc
			diffuse_inline_openacc=diffuse_${COMPILER_ID}_f_inline_openacc
	)
	target_link_libraries (
		stencil_f
			OpenACC
	)
endif()

if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI")
	target_sources (
		stencil_f PRIVATE
			laplap_cuda.cuf
			inline_cuda.cuf
	)
	target_compile_definitions (
		stencil_f PRIVATE
			diffuse_laplap_cuda=diffuse_${COMPILER_ID}_f_laplap_cuda
			diffuse_inline_cuda=diffuse_${COMPILER_ID}_f_inline_cuda
	)
	set_source_files_properties (laplap_cuda.cuf PROPERTIES LANGUAGE "Fortran")
	set_source_files_properties (laplap_cuda.cuf PROPERTIES COMPILE_OPTIONS "-Mcuda=cc60,cuda10.1")
	set_source_files_properties (inline_cuda.cuf PROPERTIES LANGUAGE "Fortran")
	set_source_files_properties (inline_cuda.cuf PROPERTIES COMPILE_OPTIONS "-Mcuda=cc60,cuda10.1")
	target_link_options (
		stencil_f PRIVATE
			-Mcuda=cc60,cuda10.1
	)
endif ()
			
set_property (
	TARGET stencil_f
	PROPERTY OUTPUT_NAME stencil_${COMPILER_ID}_f
)
install (TARGETS stencil_f)
