string (TOLOWER "${CMAKE_Fortran_COMPILER_ID}" COMPILER_ID)

add_library (
	stencil_f
		m_assert.f
		laplap_seq.f
		inline_seq.f
		laplap_openmp.f
		inline_openmp.f
		laplap_openmp_target.f
		inline_openmp_target.f
		laplap_openacc.f
		inline_openacc.f
)
target_compile_definitions (
	stencil_f PRIVATE
		diffuse_laplap_seq=diffuse_${COMPILER_ID}_f_laplap_seq
		diffuse_inline_seq=diffuse_${COMPILER_ID}_f_inline_seq
		diffuse_laplap_openmp=diffuse_${COMPILER_ID}_f_laplap_openmp
		diffuse_inline_openmp=diffuse_${COMPILER_ID}_f_inline_openmp
		diffuse_laplap_openmp_target=diffuse_${COMPILER_ID}_f_lapap_openmp_target
		diffuse_inline_openmp_target=diffuse_${COMPILER_ID}_f_inline_openmp_target
		diffuse_laplap_openacc=diffuse_${COMPILER_ID}_f_laplap_openacc
		diffuse_inline_openacc=diffuse_${COMPILER_ID}_f_inline_openacc
)
target_link_libraries (
	stencil_f
		OpenMP::Fortran
		OpenACC::Fortran
)
set_property (
	TARGET stencil_f
	PROPERTY OUTPUT_NAME stencil_${COMPILER_ID}_f
)
install (TARGETS stencil_f)
