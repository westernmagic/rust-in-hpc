use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, PlotConfiguration, AxisScale};
use gemm_perf::{rust_dgemm, blas_dgemm};

pub fn dgemm_benchmark(crit: &mut Criterion) {
    let plot_config = PlotConfiguration::default()
        .summary_scale(AxisScale::Logarithmic);
    let mut group = crit.benchmark_group("dgemm");
    group.plot_config(plot_config);

    for i in [4u16, 8u16, 16u16, 32u16, 64u16, 128u16, 256u16, 512u16].iter() {
        let (m, n, k) = (*i, *i, *i);
        let alpha = 1.0;
        let beta = 1.0;
        let a = vec![0.0f64; (m as usize) * (k as usize)];
        let b = vec![0.0f64; (k as usize) * (n as usize)];
        let mut c = vec![0.0f64; (m as usize) * (n as usize)];

        group.bench_with_input(
            BenchmarkId::new("Rust", i), i,
            |bb, _| bb.iter(|| rust_dgemm(m, n, k, alpha, &a, &b, beta, &mut c))
        );
        group.bench_with_input(
            BenchmarkId::new("BLAS", i), i,
            |bb, _| bb.iter(|| blas_dgemm(m, n, k, alpha, &a, &b, beta, &mut c))
        );
    }
    group.finish();
}

criterion_group!(benches, dgemm_benchmark);
criterion_main!(benches);
