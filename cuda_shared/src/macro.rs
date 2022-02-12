macro_rules! int_to_str {
    ($int: expr) => {
        if $int < 0 {
            concat!("-", int_to_str!(-$int))
        } else {
            if $int == 0 {
                "0"
            } else if $int == 1 {
                "1"
            } else if $int == 2 {
                "2"
            } else if $int == 3 {
                "3"
            } else if $int == 4 {
                "4"
            } else if $int == 5 {
                "5"
            } else if $int == 6 {
                "6"
            } else if $int == 7 {
                "7"
            } else if $int == 8 {
                "8"
            } else if $int == 9 {
                "9"
            } else {
                concat!(int_to_str!(($int - ($int % 10)) / 10), int_to_str!($int % 10))
            }
        }
    }
}
