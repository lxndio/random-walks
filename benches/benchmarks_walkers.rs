use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use randomwalks_lib::dp::builder::DynamicProgramBuilder;
use randomwalks_lib::dp::DynamicPrograms;
use randomwalks_lib::kernel::normal_dist::NormalDistGenerator;
use randomwalks_lib::kernel::simple_rw::SimpleRwGenerator;
use randomwalks_lib::kernel::Kernel;
use randomwalks_lib::walker::multi_step::MultiStepWalker;
use randomwalks_lib::walker::standard::StandardWalker;
use randomwalks_lib::walker::Walker;

pub fn benchmark_walker_standard(c: &mut Criterion) {
    let walk_qtys = vec![1, 10, 100, 1000, 10000, 100000, 1000000];
    let mut group = c.benchmark_group("walker_standard_vq");

    let kernel = Kernel::from_generator(SimpleRwGenerator).unwrap();
    let mut dp = DynamicProgramBuilder::new()
        .simple()
        .time_limit(400)
        .kernel(kernel.clone())
        .build()
        .unwrap();

    dp.compute();

    let walker = StandardWalker { kernel: kernel };

    for qty in walk_qtys.iter() {
        group
            .sample_size(10)
            .bench_with_input(BenchmarkId::from_parameter(qty), qty, |b, qty| {
                b.iter(|| walker.generate_paths(&dp, *qty, 100, 0, 400));
            });
    }
}

pub fn benchmark_walker_multistep(c: &mut Criterion) {
    let walk_qtys = vec![1, 10, 100, 1000, 10000, 100000, 1000000];
    let mut group = c.benchmark_group("walker_multistep_vq");

    let kernel = Kernel::from_generator(NormalDistGenerator::new(10.0, 21)).unwrap();
    let mut dp = DynamicProgramBuilder::new()
        .simple()
        .time_limit(400)
        .kernel(kernel.clone())
        .build()
        .unwrap();

    dp.compute();

    let walker = MultiStepWalker {
        max_step_size: 10,
        kernel: kernel,
    };

    for qty in walk_qtys.iter() {
        group
            .sample_size(10)
            .bench_with_input(BenchmarkId::from_parameter(qty), qty, |b, qty| {
                b.iter(|| walker.generate_paths(&dp, *qty, 100, 0, 400));
            });
    }
}

criterion_group!(
    benchmarks_walkers,
    benchmark_walker_standard,
    benchmark_walker_multistep,
);
