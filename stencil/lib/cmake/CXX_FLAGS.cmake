set (
	CMAKE_CXX_FLAGS ""
	CACHE STRING "Flags used by the CXX compiler during all build types."
	FORCE
)
set (
	CMAKE_CXX_FLAGS_DEBUG ""
	CACHE STRING "Flags used by the CXX compiler during Debug builds."
	FORCE
)
set (
	CMAKE_CXX_FLAGS_RELEASE ""
	CACHE STRING "Flags used by the CXX compiler during Release builds."
	FORCE
)
set (
	CMAKE_CXX_FLAGS_RELWITHDEBINFO ""
	CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
	FORCE
)
add_library (OpenMP::CXX  INTERFACE IMPORTED)
add_library (OpenACC::CXX INTERFACE IMPORTED)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set (
		CMAKE_CXX_FLAGS
		"-std=c++1z -D_USE_MATH_DEFINES=1 -Wall -Wextra -Wnon-virtual-dtor -Wconversion -Wcast-align -Wformat=2 -Wformat-security -Wmissing-declarations -Wstrict-overflow -Wtrampolines -Wreorder -Wsign-promo -pedantic -Wno-sign-conversion -save-temps=obj"
		CACHE STRING "Flags used by the CXX compiler during all build types."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g3 -fno-omit-frame-pointer -ftrapv"
		CACHE STRING "Flags used by the CXX compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -ffast-math -funroll-loops -fomit-frame-pointer -fopt-info" # -flto -fno-fat-lto-objects
		CACHE STRING "Flags used by the CXX compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O3 -g2 -fno-omit-frame-pointer -fopt-info-all=opt.lst" # -flto -fno-fat-lto-objects
		CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
		FORCE
	)
	set_property (TARGET OpenMP::CXX  PROPERTY INTERFACE_COMPILE_OPTIONS -fopenmp)
	set_property (TARGET OpenMP::CXX  PROPERTY INTERFACE_LINK_LIBRARIES  -fopenmp)
	set_property (TARGET OpenACC::CXX PROPERTY INTERFACE_COMPILE_OPTIONS -fopenacc)
	set_property (TARGET OpenACC::CXX PROPERTY INTERFACE_LINK_LIBRARIES  -fopenacc)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	set (
		CMAKE_CXX_FLAGS
		"-std=c++17 -D_USE_MATH_DEFINES=1 -Wall -Wextra -Wnon-virtual-dtor -Wconversion -Wformat=2 -Wformat-security -Wmissing-declarations -Woverloaded-virtual -Wreorder -Wsign-promo -pedantic"
		CACHE STRING "Flags used by the CXX compiler during all build types."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g3 -fno-omit-frame-pointer"
		CACHE STRING "Flags used by the CXX compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -unroll-aggressive -ipo -fno-fat-lto-objects -fomit-frame-pointer -qopt-prefetch -qopt-report=5 -qopt-report-annotate"
		CACHE STRING "Flags used by the CXX compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O3 -g -ipo -fno-fat-lto-objects -fno-omit-frame-pointer -qopt-prefetch -qopt-report=5 -qopt-report-annotate"
		CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
		FORCE
	)
	set_property (TARGET OpenMP::CXX PROPERTY INTERFACE_COMPILE_OPTIONS -qopenmp)
	set_property (TARGET OpenMP::CXX PROPERTY INTERFACE_LINK_LIBRARIES  -qopenmp)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
	set (
		CMAKE_CXX_FLAGS
		"-std=c++17 -march=native -D_USE_MATH_DEFINES=1 -Wall -Wextra -Wnon-virtual-dtor -Wconversion -Wcast-align -Wformat=2 -Wformat-security -Wmissing-declarations -Wstrict-overflow -Woverloaded-virtual -Wreorder -Wsign-promo -pedantic"
		CACHE STRING "Flags used by the CXX compiler during all build types."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g3 -fno-omit-frame-pointer"
		CACHE STRING "Flags used by the CXX compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -ffast-math -fomit-frame-pointer -fsave-optimization-record" # -flto
		CACHE STRING "Flags used by the CXX compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O3 -g2 -fno-omit-frame-pointer -fsave-optimization-record" # -flto
		CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
		FORCE
	)
	set_property (TARGET OpenMP::CXX PROPERTY INTERFACE_COMPILE_OPTIONS -fopenmp -fopenmp-targets=nvptx64)
	set_property (TARGET OpenMP::CXX PROPERTY INTERFACE_LINK_LIBRARIES  -fopenmp -fopenmp-targets=nvptx64)
	# set_property (TARGET OpenACC::CXX PROPERTY INTERFACE_COMPILE_OPTIONS -hacc)
	# set_property (TARGET OpenACC::CXX PROPERTY INTERFACE_LINK_LIBRARIES  -hacc)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
	set (
		CMAKE_CXX_FLAGS
		"--c++17 -Wall -pedantic"
		CACHE STRING "Flags used by the CXX compiler during all build types."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g"
		CACHE STRING "Flags used by the CXX compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -fast -Munroll -Minline -Mmovnt -Mconcur -Mipa=fast,inline -Mlist -Minfo=all"
		CACHE STRING "Flags used by the CXX compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -gopt"
		CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
		FORCE
	)
	set_property (TARGET OpenMP::CXX  PROPERTY INTERFACE_COMPILE_OPTIONS -mp=nonuma)
	set_property (TARGET OpenMP::CXX  PROPERTY INTERFACE_LINK_LIBRARIES  -mp=nonuma)
	set_property (TARGET OpenACC::CXX PROPERTY INTERFACE_COMPILE_OPTIONS -acc -ta=tesla,cc60)
	set_property (TARGET OpenACC::CXX PROPERTY INTERFACE_LINK_LIBRARIES  -acc -ta=tesla,cc60)
endif ()

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 :
