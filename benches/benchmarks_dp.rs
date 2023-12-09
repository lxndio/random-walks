use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use randomwalks_lib::dp::builder::DynamicProgramBuilder;
use randomwalks_lib::dp::DynamicPrograms;
use randomwalks_lib::kernel::simple_rw::SimpleRwGenerator;
use randomwalks_lib::kernel::Kernel;
use randomwalks_lib::kernel::normal_dist::NormalDistGenerator;

// DPs with varying time limit, SRW kernel
pub fn benchmark_dp_1(c: &mut Criterion) {
    let time_limits = (100..=600).step_by(100);
    let mut group = c.benchmark_group("DP_vt_SRW");
    group.sampling_mode(SamplingMode::Flat);

    let kernel = Kernel::from_generator(SimpleRwGenerator).unwrap();

    for time_limit in time_limits {
        group.sample_size(10).bench_with_input(
            BenchmarkId::from_parameter(time_limit),
            &time_limit,
            |b, &time_limit| {
                let mut dp = DynamicProgramBuilder::new()
                    .simple()
                    .time_limit(time_limit)
                    .kernel(kernel.clone())
                    .build()
                    .unwrap();

                b.iter(|| dp.compute());
            },
        );
    }
}

// DPs with varying time limits, NormalDist kernel size 11
pub fn benchmark_dp_2(c: &mut Criterion) {
    let time_limits = (100..=600).step_by(100);
    let mut group = c.benchmark_group("DP_vt_NormalDist_11");
    group.sampling_mode(SamplingMode::Flat);

    let kernel = Kernel::from_generator(NormalDistGenerator {
        diffusion: 5.0,
        size: 11,
    }).unwrap();

    for time_limit in time_limits {
        group.sample_size(10).bench_with_input(
            BenchmarkId::from_parameter(time_limit),
            &time_limit,
            |b, &time_limit| {
                let mut dp = DynamicProgramBuilder::new()
                    .simple()
                    .time_limit(time_limit)
                    .kernel(kernel.clone())
                    .build()
                    .unwrap();

                b.iter(|| dp.compute());
            },
        );
    }
}

// DP with varying time limits, NormalDist kernel varying sizes
pub fn benchmark_dp_3(c: &mut Criterion) {
    let time_limits = (200..=400).step_by(100);
    let kernel_sizes = (3..=21);

    for time_limit in time_limits {
        let mut group = c.benchmark_group(format!("DP_{}_NormalDist_vs", time_limit));
        group.sampling_mode(SamplingMode::Flat);

        for kernel_size in kernel_sizes.clone() {
            let kernel = Kernel::from_generator(NormalDistGenerator {
                diffusion: 5.0,
                size: kernel_size,
            }).unwrap();

            group.sample_size(10).bench_with_input(
                BenchmarkId::from_parameter(kernel_size),
                &kernel_size,
                |b, &kernel_size| {
                    let mut dp = DynamicProgramBuilder::new()
                        .simple()
                        .time_limit(400)
                        .kernel(kernel.clone())
                        .build()
                        .unwrap();

                    b.iter(|| dp.compute());
                },
            );
        }
    }
}

criterion_group!(
    benchmarks_dp,
    benchmark_dp_1,
    benchmark_dp_2,
    benchmark_dp_3,
);
