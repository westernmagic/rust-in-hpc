set (
	CMAKE_Fortran_FLAGS ""
	CACHE STRING "Flags used by the Fortran compiler during all build types."
	FORCE
)
set (
	CMAKE_Fortran_FLAGS_DEBUG
	CACHE STRING "Flags used by the Fortran compiler during Debug builds."
	FORCE
)
set (
	CMAKE_Fortran_FLAGS_RELEASE
	CACHE STRING "Flags used by the Fortran compiler during Release builds."
	FORCE
)
set (
	CMAKE_Fortran_FLAGS_RELWITHDEBINFO
	CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
	FORCE
)

if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
	set (
		CMAKE_Fortran_FLAGS
		"-std=gnu -ffree-form -ffree-line-length-none -cpp -Wall -Wextra -Wpedantic -Wsurprising -Wno-maybe-uninitialized -save-temps=obj -Wl,--enable-new-dtags,-rpath=/opt/gcc/8.3.0/snos/lib64"
		CACHE STRING "Flags used by the Fortran compiler during all build types."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g3 -fno-omit-frame-pointer -fcheck=all -ffpe-trap=invalid,zero,overflow,underflow,denormal"
		CACHE STRING "Flags used by the Fortran compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -ffast-math -funroll-loops -fomit-frame-pointer -fopt-info"
		CACHE STRING "Flags used by the Fortran compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -g2 -fno-omit-frame-pointer"
		CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
		FORCE
	)
	add_compile_options (
		"$<$<AND:$<COMPILE_LANG_AND_ID:Fortran,GNU>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<BOOL:$<TARGET_PROPERTY:INERPROCEDURAL_OPTIMIZATION>>>:-flto -fno-fat-lto-objects>"
	)
	target_compile_options (OpenMP  INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:Fortran>:-fopenmp>")
	target_link_options    (OpenMP  INTERFACE "SHELL: $<$<LINK_LANGUAGE:Fortran>:-fopenmp>")
	target_compile_options (OpenACC INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:Fortran>:-fopenacc>")
	target_link_options    (OpenACC INTERFACE "SHELL: $<$<LINK_LANGUAGE:Fortran>:-fopenacc>")
elseif ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
	set (
		CMAKE_Fortran_FLAGS
		"-stand f18 -free -warn all -fpp -Wl,--enable-new-dtags,-rpath=/opt/intel/compilers_and_libraries_2019.1.144/linux/compiler/lib/intel64"
		CACHE STRING "Flags used by the Fortran compiler during all build types."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g -debug all -check all -ftrapuv -fp-speculation safe"
		CACHE STRING "Flags used by the Fortran compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -unroll-aggressive -qopt-prefetch -qopt-report3"
		CACHE STRING "Flags used by the Fortran compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -g -debug all"
		CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
		FORCE
	)
	add_compile_options (
		"$<$<AND:$<COMPILE_LANG_AND_ID:Fortran,Intel>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<BOOL:$<TARGET_PROPERTY:INTERPROCEDURAL_OPTIMIZATION>>>:-ipo -fno-fat-lto-objects>"
	)
	target_compile_options (OpenMP INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:Fortran>:-qopenmp>")
	target_link_options    (OpenMP INTERFACE "SHELL: $<$<LINK_LANGUAGE:Fortran>:-qopenmp>")
elseif ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Cray") 
	set (
		CMAKE_Fortran_FLAGS
		"-f free -J. -ec -eC -em -ef -en -eT -eZ -eI -m1 -M7405,7418"
		CACHE STRING "Flags used by the Fortran compiler during all build types."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g -R bcds -K trap=denorm,divz,fp,inexact,inv,ovf,unf -h keep_frame_pointer" # -R p incompatible with OpenMP target
		CACHE STRING "Flags used by the Fortran compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -hfp2 -ra" # errors on -hfp3
		CACHE STRING "Flags used by the Fortran compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -G2 -h keep_frame_pointer"
		CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
		FORCE
	)
	add_compile_options (
		"$<$<AND:$<COMPILE_LANG_AND_ID:Fortran,Cray>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<BOOL:$<TARGET_PROPERTY:INTERPROCEDURAL_OPTIMIZATION>>>:-hwp -hpl=pl>"
	)
	target_compile_options (OpenMP  INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:Fortran>:-homp>")
	target_link_options    (OpenMP  INTERFACE "SHELL: $<$<LINK_LANGUAGE:Fortran>:-homp>")
	target_compile_options (OpenACC INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:Fortran>:-hacc>")
	target_link_options    (OpenACC INTERFACE "SHELL: $<$<LINK_LANGUAGE:Fortran>:-hacc>")
elseif ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI")
	set (
		CMAKE_Fortran_FLAGS
		"-Mfree -Mpreprocess -Mstandard"
		CACHE STRING "Flags used by the Fortran compiler during all build types."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g -Mbounds -Mchkptr -Ktrap=divz,fp,inexact,inv,ovf,unf -Mframe"
		CACHE STRING "Flags used by the Fortran compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -fast -Mnoframe -Munroll -Minline -Mmovnt -Mlist"
		CACHE STRING "Flags used by the Fortran compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -gopt -Mframe"
		CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
		FORCE
	)
	add_compile_options (
		"$<$<AND:$<COMPILE_LANG_AND_ID:Fortran,PGI>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<BOOL:$<TARGET_PROPERTY:INTERPROCEDURAL_OPTIMIZATION>>>:-Mipa=fast,inline>"
	)
	target_compile_options (OpenMP  INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:Fortran>:-mp=nonuma>")
	target_link_options    (OpenMP  INTERFACE "SHELL: $<$<LINK_LANGUAGE:Fortran>:-mp=nonuma>")
	target_compile_options (OpenACC INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:Fortran>:-acc -ta=tesla,cc60>")
	target_link_options    (OpenACC INTERFACE "SHELL: $<$<LINK_LANGUAGE:Fortran>:-acc -ta=tesla,cc60>")
endif ()

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 :
