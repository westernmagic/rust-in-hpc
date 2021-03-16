include (CMakeTestCompilerCommon)

function (cmake_test_rust_compiler)
	PrintTestCompilerStatus ("Rust")
	file (
		WRITE
		${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeTmp/testRustCompiler.rs
		"fn main() {}"
	)

	try_compile (
		CMAKE_Rust_COMPILER_WORKS
		${CMAKE_BINARY_DIR}
		SOURCES ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeTmp/testRustCompiler.rs
		OUTPUT_VARIABLE CMAKE_Rust_COMPILER_OUTPUT
	)

	if (NOT CMAKE_Rust_COMPILER_WORKS)
		PrintTestCompilerResult(CHECK_FAIL "broken")
		file (
			APPEND
			${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
			"Determining if the Rust compiler works failed "
			"with the following output:\n${CMAKE_Rust_COMPILER_OUTPUT}\n\n"
		)
		string (
			REPLACE
			"\n"
			"\n  "
			_output
			${CMAKE_Rust_COMPILER_OUTPUT}
		)
		message (
			FATAL_ERROR
			"The Rust compiler\n  \"${CMAKE_Rust_COMPILER}\"\n"
			"is not able to compile a simple test program.\n It fails "
			"with the following output:\n  ${_output}\n\n"
			"CMake will not be able to properly generate this project."
		)
	endif ()

	PrintTestCompilerResult(CHECK_PASS "works")
	file (
		APPEND
		${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
		"Determining if the Rust compiler works passed with "
		"the following output:\n${CMAKE_Rust_COMPILER_OUTPUT}\n\n"
	)

	configure_file (
		${CMAKE_CURRENT_LIST_DIR}/CMakeRustCompiler.cmake.in
		${CMAKE_PLATFORM_INFO_DIR}/CMakeRustCompiler.cmake
		@ONLY
	)
endfunction ()

cmake_test_rust_compiler ()
