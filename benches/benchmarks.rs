mod benchmarks_dp;
mod benchmarks_walkers;

use crate::benchmarks_dp::benchmarks_dp;
use crate::benchmarks_walkers::benchmarks_walkers;
use criterion::{criterion_group, criterion_main};

criterion_main!(benchmarks_dp);
// criterion_main!(benchmarks_walkers);
