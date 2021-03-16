#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use std::ffi::CString;

#[inline]
pub fn region_begin(label: &str) -> Result<(), ()> {
	unimplemented!();

	let id = 1;
	assert!(id > 0);
	match unsafe { PAT_region_begin(id, CString::new(label).unwrap().as_ptr()) as u32 } {
		PAT_API_OK => Ok(()),
		PAT_API_FAIL => Err(()),
		_ => panic!()
	}
}

#[inline]
pub fn region_end() -> Result<(), ()> {
	unimplemented!();

	let id = 1;
	assert!(id > 0);
	match unsafe { PAT_region_end(id) as u32 } {
		PAT_API_OK => Ok(()),
		PAT_API_FAIL => Err(()),
		_ => panic!()
	}
}

#[inline]
pub fn record(b: bool) -> bool {
	let state = if b { PAT_STATE_ON } else { PAT_STATE_OFF } as i32;
	match unsafe { PAT_record(state) as u32 } {
		PAT_STATE_ON => true,
		PAT_STATE_OFF => false,
		_ => panic!()
	}
}

#[inline]
pub fn state() -> bool {
	match unsafe { PAT_record(PAT_STATE_QUERY as i32) as u32 } {
		PAT_STATE_ON => true,
		PAT_STATE_OFF => false,
		_ => panic!()
	}
}

#[inline]
pub fn flush_buffer() -> Result<u64, ()> {
	let mut nbytes = 0;
	match unsafe { PAT_flush_buffer(&mut nbytes) as u32 } {
		PAT_API_OK => Ok(nbytes),
		PAT_API_FAIL => Err(()),
		_ => panic!()
	}
}

#[inline]
pub fn counters() {
	unimplemented!();
}
