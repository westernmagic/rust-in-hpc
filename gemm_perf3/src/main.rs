use ndarray::{Array2, ArrayView2, Axis};

fn mm_seq<'a>(
    a: &ArrayView2<'a, f64>,
    b: &ArrayView2<'a, f64>,
) -> Array2<f64> {
    a.dot(b)
}

fn mm_par<'a>(
    a: &ArrayView2<'a, f64>,
    b: &ArrayView2<'a, f64>
) -> Array2<f64> {
    const CHUNK_SIZE: usize = 16;

    let mut aa = Vec::new();
    for chunk in a.axis_chunks_iter(Axis(1), CHUNK_SIZE) {
        for chunk in chunk.axis_chunks_iter(Axis(0), CHUNK_SIZE) {
            aa.push(chunk);
        }
    }

    a.dot(b)
}

fn main() {
    println!("Hello, world!");
}
