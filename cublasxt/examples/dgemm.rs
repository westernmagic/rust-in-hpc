use cublasxt::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (m, n, k) = (2, 4, 3);
    let a = vec![
        1.0, 4.0,
        2.0, 5.0,
        3.0, 6.0,
    ];
    let b = vec![
        1.0, 5.0,  9.0,
        2.0, 6.0, 10.0,
        3.0, 7.0, 11.0,
        4.0, 8.0, 12.0,
    ];
    let mut c = vec![
        2.0, 7.0,
        6.0, 2.0,
        0.0, 7.0,
        4.0, 2.0,
    ];

    let context = Context::new()?;
    context.dgemm(Op::N, Op::N, m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m)?;

    assert_eq!(
        c, vec![
            40.0,  90.0,
            50.0, 100.0,
            50.0, 120.0,
            60.0, 130.0,
        ]
    );

    Ok(())
}
