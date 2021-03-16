include (CMakeLanguageInformation)

# if (UNIX)
# 	set (CMAKE_Rust_OUTPUT_EXTENSION .o)
# else ()
# 	set (CMAKE_RUST_OUTPUT_EXTENSION .obj)
# endif ()

set (CMAKE_Rust_OUTPUT_EXTENSION .rlink)

if (NOT CMAKE_Rust_CREATE_SHARED_LIBRARY)
	set (
		CMAKE_Rust_CREATE_SHARED_LIBRARY
		"<CMAKE_Rust_COMPILER> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <LINK_LIBRARIES> -Z link-only --crate-type cdylib --crate-name <TARGET_BASE> -o <TARGET> CMakeFiles/<TARGET_BASE>.dir/<TARGET_BASE>.rlink"
	)
endif ()

if (NOT CMAKE_Rust_CREATE_SHARED_MODULE)
	set (
		CMAKE_Rust_CREATE_SHARED_MODULE
		"${CMAKE_Rust_CREATE_SHARED_LIBRARY}"
	)
endif ()

if (NOT CMAKE_Rust_CREATE_STATIC_LIBRARY)
	set (
		CMAKE_Rust_CREATE_STATIC_LIBRARY
		"<CMAKE_Rust_COMPILER> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <LINK_LIBRARIES> -Z link-only --crate-type staticlib --crate-name <TARGET_BASE> -o <TARGET> CMakeFiles/<TARGET_BASE>.dir/<TARGET_BASE>.rlink"
	)
endif ()

if (NOT CMAKE_Rust_COMPILE_OBJECT)
	set (
		CMAKE_Rust_COMPILE_OBJECT
		"<CMAKE_Rust_COMPILER> <FLAGS> -Z no-link --crate-name <TARGET_BASE> --out-dir CMakeFiles/<TARGET_BASE>.dir <SOURCE>"
	)
endif ()

if (NOT CMAKE_Rust_LINK_EXECUTABLE)
	set (
		CMAKE_Rust_LINK_EXECUTABLE
		"<CMAKE_Rust_COMPILER> <FLAGS> <LINK_FLAGS> <LINK_LIBRARIES> -Z link-only --crate-type bin --crate-name <TARGET_BASE> -o <TARGET> CMakeFiles/<TARGET_BASE>.dir/<TARGET_BASE>.rlink"
	)
endif ()

set (CMAKE_Rust_INFORMATION_LOADED 1)
