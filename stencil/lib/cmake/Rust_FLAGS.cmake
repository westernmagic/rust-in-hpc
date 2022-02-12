set (
	CARGO_RUST_FLAGS "-C target-cpu=haswell"
	CACHE STRING "Flags to pass to the Rust compiler during all build types."
	FORCE
)
set (
	CARGO_RUST_FLAGS_DEBUG "-C opt-level=0 -C debug-assertions=yes -C debuginfo=2 -C force-frame-pointers=yes -C codegen-units=16 -C incremental=true"
	CACHE STRING "Flags to pass to the Rust compiler during Debug builds."
	FORCE
)
set (
	CARGO_RUST_FLAGS_RELEASE "-C opt-level=3 -C debug-assertions=yes -C debuginfo=no -C force-frame-pointers=no -C codegen-units=1 -C incremental=false" # -C remark=all
	CACHE STRING "Flags to pass to the Rust compiler during Release builds."
	FORCE
)
set (
	CARGO_RUST_FLAGS_RELWITHDEBINFO "-C opt-level=3 -C debuginfo=2 -C debug-assertions=no -C force-frame-pointers=yes -C codegen-units=1 -C incremental=false" # -C remark=all
	CACHE STRING "Flags to pass to the Rust compiler during ReWithDebInfo builds."
	FORCE
)
