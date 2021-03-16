#[macro_export]
macro_rules! print {
    ($($arg: tt)*) => {
        let msg = $crate::format!($($arg)*);
        #[allow(unused_unsafe)]
        unsafe { ::core::arch::nvtpx::vprintf(msg.as_ptr(), ::core::ptr::null_mut()); }
    }
}

#[macro_export]
macro_rules! println {
    () => { println!("") };
    ($fmt: expr) => { $crate::print!(concat!($fmt, "\n")) };
    ($fmt: expr,) => { println!($fmt) };
    ($fmt: expr, $($arg: tt)*) => { $crate::print!(concat!($fmt, "\n"), $($arg)*) };
}

#[macro_export]
macro_rules! assert {
    ($cond: expr) => { assert!($cond, "\nassertion failed: {}", stringify!($expr)) };
    ($cond: expr,) => { assert!($cond) };
    ($cond: expr, $($arg: tt)+) => {
        if !expr {
            let msg = $crate::format!($($arg)*);
            unsafe {
                ::core::arch::nvptx::__assert_fail(msg.as_ptr(), file!().as_ptr(), line!(), "".as_ptr())
            }
        }
    };
}

#[macro_export]
macro_rules! assert_eq {
    ($left: expr, $right: expr) => {
        assert!(
            "\nassertion failed: `({} == {})`\n  left: `{:?}`,\n right: `{:?}`",
            stringify!($left),
            stringify!($right),
            $left,
            $right
        )
    };
    ($left: expr, $right: expr,) => { assert_eq!($left, $right) };
    ($left: expr, $right: expr, $($arg: tt)+) => {
        assert!(
            concat!(
                $crate::format!(
                    "\nassertion failed: `({} == {})`\n  left: `{:?}`,\n right: `{:?}`",
                    stringify!($left),
                    stringify!($right),
                    $left,
                    $right
                ),
                ": ",
                $crate::format!($($arg)*)
            )
        )
    };
}

#[macro_export]
macro_rules! assert_ne {
    ($left: expr, $right: expr) => {
        assert!(
            "\nassertion failed: `({} != {})`\n  left: `{:?}`,\n right: `{:?}`",
            stringify!($left),
            stringify!($right),
            $left,
            $right
        )
    };
    ($left: expr, $right: expr,) => { assert_ne!($left, $right) };
    ($left: expr, $right: expr, $($arg: tt)*) => {
        assert!(
            concat!(
                $crate::format!(
                    "\nassertion failed: `({} != {})`\n  left: `{:?}`\n right: `{:?}`",
                    stringify!($left),
                    stringify!($right),
                    $left,
                    $right
                ),
                ": ",
                $crate::format!($($arg)*)
            )
        )
    }
}

#[macro_export]
macro_rules! debug_assert {
    ($($arg: tt)*) => { if cfg!(debug_assertions) { $crate::assert!($($arg)*) } };
}

#[macro_export]
macro_rules! debug_assert_eq {
    ($($arg: tt)*) => { if cfg!(debug_assertions) { $crate::assert_eq!($($arg)*) } };
}

#[macro_export]
macro_rules! debug_assert_ne {
    ($($arg: tt)*) => { if cfg!(debug_assertions) { $crate::assert_ne!($($arg)*) } };
}
