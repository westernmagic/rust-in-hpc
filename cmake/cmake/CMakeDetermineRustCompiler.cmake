if (NOT CMAKE_Rust_COMPILER)
	find_package(Rust COMPONENTS RUSTC)

	if (Rust_FOUND)
		set (CMAKE_Rust_COMPILER "${RUSTC_EXECUTABLE}")
		set (CMAKE_Rust_COMPILER_ID "Rust")
		set (CMAKE_Rust_COMPILER_VERSION "${RUSTC_VERSION}")
	endif ()
endif ()

if (CMAKE_Rust_COMPILER)
	set (CMAKE_Rust_COMPILER_LOADED 1)
endif ()

set (CMAKE_Rust_COMPILER_ENV_VAR "RUSTC")

configure_file (
	${CMAKE_CURRENT_LIST_DIR}/CMakeRustCompiler.cmake.in
	# ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${CMAKE_VERSION}/CMakeRustCompiler.cmake
	${CMAKE_PLATFORM_INFO_DIR}/CMakeRustCompiler.cmake
	@ONLY
)
