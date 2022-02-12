set (
        CMAKE_CXX_FLAGS ""
        CACHE STRING "Flags used by the CXX compiler during all build types."
        FORCE
)
set (
        CMAKE_CXX_FLAGS_DEBUG
        CACHE STRING "Flags used by the CXX compiler during Debug builds."
        FORCE
)
set (
        CMAKE_CXX_FLAGS_RELEASE
        CACHE STRING "Flags used by the CXX compiler during Release builds."
        FORCE
)
set (
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
        FORCE
)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set (
		CMAKE_CXX_FLAGS
		"-std=c++14 -D_USE_MATH_DEFINES=1 -Wall -Wextra -Wnon-virtual-dtor -Wconversion -Wcast-align -Wformat=2 -Wformat-security -Wmissing-declarations -Wstrict-overflow -Wtrampolines -Wreorder -Wsign-promo -pedantic -Wno-sign-conversion -save-temps=obj"
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
		"-UDEBUG -DNDEBUG=1 -O3 -ffast-math -funroll-loops -fomit-frame-pointer -fopt-info"
		CACHE STRING "Flags used by the CXX compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -g2 -fno-omit-frame-pointer"
		CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
		FORCE
	)
	add_compile_options (
		"$<$<AND:$<COMPILE_LANG_AND_ID:CXX,GNU>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<BOOL:$<TARGET_PROPERTY:INTERPROCEDURAL_OPTIMIZATION>>>:-flto -fno-fat-lto-objects>"
	)
	target_compile_options (OpenMP  INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>")
	target_link_options    (OpenMP  INTERFACE "SHELL: $<$<LINK_LANGUAGE:CXX>:-fopenmp>")
	target_compile_options (OpenACC INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CXX>:-fopenacc>")
	target_link_options    (OpenACC INTERFACE "SHELL: $<$<LINK_LANGUAGE:CXX>:-fopenacc>")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	set (
		CMAKE_CXX_FLAGS
		"-std=c++17 -D_USE_MATH_DEFINES=1 -Wall -Wextra -Wnon-virtual-dtor -Wconversion -Wformat=2 -Wformat-security -Wmissing-declarations -Woverloaded-virtual -Wreorder -Wsign-promo -pedantic -Wl,--enable-new-dtags,-rpath=/opt/intel/compilers_and_libraries_2019.1.144/linux/compiler/lib/intel64"
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
		"-UDEBUG -DNDEBUG=1 -O3 -unroll-aggressive -qopt-prefetch -qopt-report -fomit-frame-pointer"
		CACHE STRING "Flags used by the CXX compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -g -fno-omit-frame-pointer"
		CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
		FORCE
	)
	add_compile_options (
		"$<$<AND:$<COMPILE_LANG_AND_ID:CXX,Intel>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<BOOL:$<TARGET_PROPERTY:INTERPROCEDURAL_OPTIMIZATION>>>:-ipo -fno-fat-lto-objects>"
	)
	target_compile_options (OpenMP INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CXX>:-qopenmp>")
	target_link_options    (OpenMP INTERFACE "SHELL: $<$<LINK_LANGUAGE:CXX>:-qopenmp>")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") 
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
		"-UDEBUG -DNDEBUG=1 -O3 -ffast-math -fomit-frame-pointer -fsave-loopmark"
		CACHE STRING "Flags used by the CXX compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O3 -g2 -fno-omit-frame-pointer -fsave-loopmark"
		CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
		FORCE
	)
	add_compile_options (
		"$<$<AND:$<COMPILE_LANG_AND_ID:CXX,Clang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<BOOL:$<TARGET_PROPERTY:INTERPROCEDURAL_OPTIMIZATION>>>:-flto>"
	)
	target_compile_options (OpenMP INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CXX>:-fopenmp -fopenmp-targets=nvptx64>")
	target_link_options    (OpenMP INTERFACE "SHELL: $<$<LINK_LANGUAGE:CXX>:-fopenmp -fopenmp-targets=nvptx64>")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
	set (
		CMAKE_CXX_FLAGS
		"--c++17 -Wall -pedantic"
		CACHE STRING "Flags used by the CXX compiler during all build types."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g -Mframe"
		CACHE STRING "Flags used by the CXX compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -fast -Mnoframe -Munroll -Minline -Mmovnt -Mlist"
		CACHE STRING "Flags used by the CXX compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -gopt -Mframe"
		CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
		FORCE
	)
	add_compile_options (
		"$<$<AND:$<COMPILE_LANG_AND_ID:CXX,PGI>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>,$<BOOL:$<TARGET_PROPERTY:INTERPROCEDURAL_OPTIMIZATION>>>:-Mipa=fast,inline>"
	)
	target_compile_options (OpenMP  INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CXX>:-mp=nonuma>")
	target_link_options    (OpenMP  INTERFACE "SHELL: $<$<LINK_LANGUAGE:CXX>:-mp=nonuma>")
	target_compile_options (OpenACC INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CXX>:-acc -ta=tesla,cc60,cuda10.1>")
	target_link_options    (OpenACC INTERFACE "SHELL: $<$<LINK_LANGUAGE:CXX>:-acc -ta=tesla,cc60,cuda10.1>")
endif ()

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 :
