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
add_library (OpenMP::Fortran  INTERFACE IMPORTED)
add_library (OpenACC::Fortran INTERFACE IMPORTED)

if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
	set (
		CMAKE_Fortran_FLAGS
		"-std=gnu -ffree-form -ffree-line-length-none -cpp -Wall -Wextra -Wpedantic -Wsurprising -Wno-maybe-uninitialized -save-temps=obj"
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
		"-UDEBUG -DNDEBUG=1 -O3 -ffast-math -funroll-loops -fomit-frame-pointer -fopt-info-all=opt.lst" # -flto -fno-fat-lto-objects
		CACHE STRING "Flags used by the Fortran compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O3 -g2 -fno-omit-frame-pointer -fopt-info-all=opt.lst" # -flto -fno-fat-lto-objects
		CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
		FORCE
	)
	set_property (TARGET OpenMP::Fortran  PROPERTY INTERFACE_COMPILE_OPTIONS -fopenmp)
	set_property (TARGET OpenMP::Fortran  PROPERTY INTERFACE_LINK_LIBRARIES  -fopenmp)
	set_property (TARGET OpenACC::Fortran PROPERTY INTERFACE_COMPILE_OPTIONS -fopenacc)
	set_property (TARGET OpenACC::Fortran PROPERTY INTERFACE_LINK_LIBRARIES  -fopenacc)
elseif ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
	set (
		CMAKE_Fortran_FLAGS
		"-stand f18 -free -warn all -fpp"
		CACHE STRING "Flags used by the Fortran compiler during all build types."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g -debug all -check all -ftrapuv -fp-speculation safe -fno-omit-frame-pointer"
		CACHE STRING "Flags used by the Fortran compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -no-rec-div -fp-model fast=2 -unroll-aggressive -fomit-frame-pointer -qopt-prefetch -qopt-report=5 -qopt-report-annotate" # -ipo -fno-fat-lto-objects
		CACHE STRING "Flags used by the Fortran compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O3 -no-prec-div -fp-model fast=2 -g -fno-omit-frame-pointer -qopt-prefetch -qopt-report=5 -qopt-report-annotate" # -ipo -fno-fat-lto-objects
		CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
		FORCE
	)
	set_property (TARGET OpenMP::Fortran PROPERTY INTERFACE_COMPILE_OPTIONS -qopenmp)
	set_property (TARGET OpenMP::Fortran PROPERTY INTERFACE_LINK_LIBRARIES  -qopenmp)
elseif ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Cray") 
	set (
		CMAKE_Fortran_FLAGS
		"-f free -J. -ec -eC -em -ef -en -eT -eZ -eI -m1 -M7405,7418"
		CACHE STRING "Flags used by the Fortran compiler during all build types."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -G0 -eD -h develop -R bcdps -K trap=denorm,divz,fp,inexact,inv,ovf,unf"
		CACHE STRING "Flags used by the Fortran compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -hfp2 -ra" # TODO errors on -hfp3
		CACHE STRING "Flags used by the Fortran compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O3 -hfp2 -G2 -ra" # TODO
		CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
		FORCE
	)
	set_property (TARGET OpenMP::Fortran  PROPERTY INTERFACE_COMPILE_OPTIONS -homp)
	set_property (TARGET OpenMP::Fortran  PROPERTY INTERFACE_LINK_LIBRARIES  -homp)
	set_property (TARGET OpenACC::Fortran PROPERTY INTERFACE_COMPILE_OPTIONS -hacc)
	set_property (TARGET OpenACC::Fortran PROPERTY INTERFACE_LINK_LIBRARIES  -hacc)

elseif ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI")
	set (
		CMAKE_Fortran_FLAGS
		"-Mfree -Mpreprocess -Mstandard"
		CACHE STRING "Flags used by the Fortran compiler during all build types."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g -Mbounds -Mchkptr -Ktrap=divz,fp,inexact,inv,ovf,unf"
		CACHE STRING "Flags used by the Fortran compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -fast -Munroll -Minline -Mmovnt -Mconcur -Mlist -Minfo=all -Mneginfo=all" # -Mipa=fast,inline
		CACHE STRING "Flags used by the Fortran compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O3 -gopt -Mlist -Minfo=all -Mneginfo=all" # TODO
		CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
		FORCE
	)
	set_property (TARGET OpenMP::Fortran  PROPERTY INTERFACE_COMPILE_OPTIONS -mp=nonuma)
	set_property (TARGET OpenMP::Fortran  PROPERTY INTERFACE_LINK_LIBRARIES  -mp=nonuma)
	set_property (TARGET OpenACC::Fortran PROPERTY INTERFACE_COMPILE_OPTIONS -acc -ta=tesla,cc60)
	set_property (TARGET OpenACC::Fortran PROPERTY INTERFACE_LINK_LIBRARIES  -acc -ta=tesla,cc60)
endif ()

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 :
