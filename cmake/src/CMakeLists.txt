add_executable (main main.rs)
add_library (main_shared SHARED main.rs)
add_library (main_static STATIC main.rs)
