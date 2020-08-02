string (TOLOWER "${CMAKE_Fortran_COMPILER_ID}" COMPILER_ID)

add_library (
	stencil_f
		m_assert.f
		v0_base.f
		v1_inline.f
		v2_openmp.f
		v3_openmp.f
)
target_compile_definitions (
	stencil_f PRIVATE
		diffuse_v0_base=diffuse_${COMPILER_ID}_f_v0_base
		diffuse_v1_inline=diffuse_${COMPILER_ID}_f_v1_inline
		diffuse_v2_openmp=diffuse_${COMPILER_ID}_f_v2_openmp
		diffuse_v3_openmp=diffuse_${COMPILER_ID}_f_v3_openmp
)
target_link_libraries (
	stencil_f
		OpenMP::Fortran
)
set_property (
	TARGET stencil_f
	PROPERTY OUTPUT_NAME stencil_${COMPILER_ID}_f
)
install (TARGETS stencil_f)
