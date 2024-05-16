use std::borrow::BorrowMut;
use std::fmt::Debug;
use std::fs;
use std::ops::{DerefMut, Range};
use std::path::{Path, PathBuf};
use std::sync::mpsc::channel;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use anyhow::{bail, Context};
use log::{debug, trace};
use num::traits::ToBytes;
use num::Zero;
#[cfg(feature = "plotting")]
use plotters::prelude::*;
use rayon::prelude::*;
use workerpool::thunk::{Thunk, ThunkWorker};
use workerpool::Pool;
#[cfg(feature = "saving")]
use {
    std::fs::File,
    std::io::{BufReader, Read},
    std::io::{BufWriter, Write},
    zstd::{Decoder, Encoder},
};

use crate::dp::builder::DynamicProgramBuilder;
use crate::dp::{DynamicProgramPool, DynamicPrograms};
use crate::kernel;
use crate::kernel::DirKernel;
use crate::kernel::Kernel;

#[derive(Clone)]
pub struct CorDynamicProgram {
    pub/*(crate)*/ table: Vec<Vec<Vec<Vec<f64>>>>,
    pub/*(crate)*/ time_limit: usize,
    pub/*(crate)*/ num_directions: usize,
    pub/*(crate)*/ kernels: Vec<Kernel>,
    pub/*(crate)*/ field_types: Vec<Vec<usize>>,
    pub/*(crate)*/ dir_kernel: DirKernel,
}

impl CorDynamicProgram {
    pub fn at(&self, x: isize, y: isize, d: usize, t: usize) -> f64 {
        let x = (self.time_limit as isize + x) as usize;
        let y = (self.time_limit as isize + y) as usize;

        self.table[t][d][x][y]
    }

    pub fn at_or(&self, x: isize, y: isize, d: usize, t: usize, default: f64) -> f64 {
        let (limit_neg, limit_pos) = self.limits();

        if x >= limit_neg && x <= limit_pos && y >= limit_neg && y <= limit_pos {
            let x = (self.time_limit as isize + x) as usize;
            let y = (self.time_limit as isize + y) as usize;

            self.table[t][d][x][y]
        } else {
            default
        }
    }

    pub fn set(&mut self, x: isize, y: isize, d: usize, t: usize, val: f64) {
        let x = (self.time_limit as isize + x) as usize;
        let y = (self.time_limit as isize + y) as usize;

        self.table[t][d][x][y] = val;
    }

    fn apply_kernel_at(&mut self, x: isize, y: isize, d: usize, t: usize) {
        let kernel = self.kernels[d].clone();

        // Is this important?
        let dir_kernel = self.dir_kernel.clone();

        let ks = (kernel.size() / 2) as isize;
        let (limit_neg, limit_pos) = self.limits();
        let mut sum = 0.0;

        for di in 0..self.num_directions as usize {
            
            for (prev_kernel_x, prev_kernel_y) in dir_kernel.cells_pointing_to(d) {

                let i = x + prev_kernel_x;
                let j = y + prev_kernel_y;

                if i < limit_neg || i > limit_pos || j < limit_neg || j > limit_pos {
                    continue;
                }

                let kernel_x = x - i;
                let kernel_y = y - j;

                sum += self.at(i, j, di, t-1) * self.kernels[di].prob_at(kernel_x, kernel_y);
            }
        }

        self.set(x, y, d, t, sum);
    }

    fn field_type_at(&self, x: isize, y: isize) -> usize {
        let x = (self.time_limit as isize + x) as usize;
        let y = (self.time_limit as isize + y) as usize;

        self.field_types[x][y]
    }

    fn field_type_set(&mut self, x: isize, y: isize, val: usize) {
        let x = (self.time_limit as isize + x) as usize;
        let y = (self.time_limit as isize + y) as usize;

        self.field_types[x][y] = val;
    }

    pub fn get_num_directions(&self) -> usize {
        self.num_directions
    }

    #[cfg(feature = "saving")]
    pub fn load(filename: String, kernels: Vec<Kernel>, dir_kernel: DirKernel) -> anyhow::Result<DynamicProgramPool> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let mut decoder = Decoder::new(reader).context("could not create decoder")?;

        let mut time_limit = [0u8; 8];
        let time_limit = match decoder.read_exact(&mut time_limit) {
            Ok(()) => u64::from_le_bytes(time_limit),
            Err(_) => bail!("could not read time limit from file"),
        } as usize;
        let mut num_directions = [0u8; 8];
        let num_directions = match decoder.read_exact(&mut num_directions) {
            Ok(()) => u64::from_le_bytes(num_directions),
            Err(_) => bail!("could not read num_directions from file"),
        };

        
        let mut dp = CorDynamicProgram {
            table: vec![
                vec![
                    vec![vec![0.0; 2 * time_limit + 1]; 2 * time_limit + 1];
                    16
                ];
                time_limit + 1
            ],
            time_limit,
            num_directions: 16,
            kernels,
            field_types: vec![vec![0; 2 * time_limit + 1]; 2 * time_limit + 1],
            dir_kernel,
        };

        // let DynamicProgramPool::Single(mut dp) = DynamicProgramBuilder::new()
        //     .simple()
        //     .time_limit(time_limit as usize)
        //     .kernel(kernel!(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        //     .build()?
        // else {
        //     unreachable!();
        // };

        let (limit_neg, limit_pos) = dp.limits();
        let mut buf = [0u8; 8];

        for t in 0..=limit_pos as usize {
            for d in 0..num_directions as usize {
                for x in limit_neg..=limit_pos {
                    for y in limit_neg..=limit_pos {
                        decoder.read_exact(&mut buf)?;
                        dp.set(x, y, d, t, f64::from_le_bytes(buf));
                    }
                }
            }
        }

        for x in limit_neg..=limit_pos {
            for y in limit_neg..=limit_pos {
                decoder.read_exact(&mut buf)?;
                dp.field_type_set(x, y, u64::from_le_bytes(buf) as usize);
            }
        }

        Ok(DynamicProgramPool::Single(dp))
    }


    // pub fn into_iter(self) -> DynamicProgramLayerIterator {
    //     DynamicProgramLayerIterator {
    //         last_layer: Vec::new(),
    //         layer: 0,
    //         dp: None,
    //         time_limit: self.time_limit,
    //         kernels: self.kernels,
    //         field_types: self.field_types,
    //     }
    // }
}

impl DynamicPrograms for CorDynamicProgram {
    #[cfg(not(tarpaulin_include))]
    fn limits(&self) -> (isize, isize) {
        (-(self.time_limit as isize), self.time_limit as isize)
    }

    fn compute(&mut self) {
        let (limit_neg, limit_pos) = self.limits();

        self.set(0, 0, 0, 0, 1.0);

        let start = Instant::now();

        for t in 1..=limit_pos as usize {
            if t % 10 == 0 {
                println!("t: {t}");
            }

            for d in 0..self.num_directions as usize {
                for x in limit_neg..=limit_pos {
                    for y in limit_neg..=limit_pos {
                        self.apply_kernel_at(x, y, d, t);
                    }
                }
            }
        }

        let duration = start.elapsed();

        println!("Computation took {:?}", duration);
    }

    fn compute_parallel(&mut self) {
        // let (limit_neg, limit_pos) = self.limits();
        // let kernels = Arc::new(RwLock::new(self.kernels.clone()));
        // let field_types = Arc::new(RwLock::new(self.field_types.clone()));
        // let pool = Pool::<ThunkWorker<(Range<isize>, Range<isize>, Vec<Vec<f64>>)>>::new(10);
        // let (tx, rx) = channel();

        // // Define chunks

        // let chunk_size = ((self.time_limit + 1) / 3) as isize;
        // let mut ranges = Vec::new();

        // for i in 0..3 - 1 {
        //     ranges.push((limit_neg + i * chunk_size..limit_neg + (i + 1) * chunk_size));
        // }

        // ranges.push(limit_neg + 2 * chunk_size..limit_pos + 1);
        // let mut chunks = Vec::new();

        // for x in 0..3 {
        //     for y in 0..3 {
        //         chunks.push((ranges[x].clone(), ranges[y].clone()));
        //     }
        // }

        // self.set(0, 0, 0, 1.0);

        // let start = Instant::now();

        // for t in 1..=limit_pos as usize {
        //     let table_old = Arc::new(RwLock::new(self.table[t - 1].clone()));

        //     for (x_range, y_range) in chunks.clone() {
        //         let kernels = kernels.clone();
        //         let field_types = field_types.clone();
        //         let table_old = table_old.clone();

        //         pool.execute_to(
        //             tx.clone(),
        //             Thunk::of(move || {
        //                 let mut probs = vec![vec![0.0; y_range.len()]; x_range.len()];
        //                 let (mut i, mut j) = (0, 0);

        //                 for x in x_range.clone() {
        //                     for y in y_range.clone() {
        //                         probs[i][j] = apply_kernel(
        //                             &table_old.read().unwrap(),
        //                             &kernels.read().unwrap(),
        //                             &field_types.read().unwrap(),
        //                             (limit_neg, limit_pos),
        //                             x,
        //                             y,
        //                         );

        //                         j += 1;
        //                     }

        //                     i += 1;
        //                     j = 0;
        //                 }

        //                 (x_range.clone(), y_range.clone(), probs)
        //             }),
        //         );
        //     }

        //     for (x_range, y_range, probs) in rx.iter().take(9) {
        //         let (mut i, mut j) = (0, 0);

        //         for x in x_range.clone() {
        //             for y in y_range.clone() {
        //                 self.table[t][(self.time_limit as isize + x) as usize]
        //                     [(self.time_limit as isize + y) as usize] = probs[i][j];

        //                 j += 1;
        //             }

        //             i += 1;
        //             j = 0;
        //         }
        //     }
        // }

        // let duration = start.elapsed();

        // println!("Computation took {:?}", duration);
    }

    #[cfg(not(tarpaulin_include))]
    fn field_types(&self) -> Vec<Vec<usize>> {
        self.field_types.clone()
    }

    #[cfg(not(tarpaulin_include))]
    #[cfg(feature = "plotting")]
    fn heatmap(&self, path: String, d: usize, t: usize) -> anyhow::Result<()> {
        let multiplier = 0.075;
        let table = self.table[t].clone();
        let size = table.len();

        let drawing_area = BitMapBackend::new(&path, (1000, 1000)).into_drawing_area();

        drawing_area.fill(&WHITE).unwrap();

        let mut ctx = ChartBuilder::on(&drawing_area)
            .build_cartesian_2d(0.0..size as f64 + 1.0, 0.0..size as f64 + 1.0)
            .unwrap();

        // ctx.configure_mesh().draw().unwrap();

        let min_prob = table
            .iter()
            .flatten()
            .flatten()
            .filter(|x| x > &&0.0)
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_prob = table
            .iter()
            .flatten()
            .flatten()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        let mut data = Vec::new();
        for i in 0..size {
            for j in 0..size {
                let value = table[d][i][j].powf(multiplier);
                let value = if value != 0.0 {
                    1.0 - (value - min_prob) / (max_prob - min_prob)
                } else {
                    1.0
                };

                data.push(((i as f64, j as f64), value));
            }
        }

        ctx.draw_series(data.iter().map(|((x, y), value)| {
            let color = RGBColor(
                (255.0 * value) as u8,
                (255.0 * value) as u8,
                (255.0 * value) as u8,
            );

            Rectangle::new([(*x, *y), (*x + 1.0, *y + 1.0)], color.filled())
        }))
        .unwrap();

        ctx.draw_series(vec![Rectangle::new([(0.0, 0.0), (1.0, 1.0)], RED.filled())])
            .unwrap();

        Ok(())
    }

    #[cfg(not(tarpaulin_include))]
    fn print(&self, d:usize, t: usize) {
        for y in 0..2 * self.time_limit + 1 {
            for x in 0..2 * self.time_limit + 1 {
                print!("{:.4} ", self.table[t][d][x][y]);
            }

            println!();
        }
    }

    // #[cfg(feature = "saving")]
    fn save(&self, filename: String) -> std::io::Result<()> {
        let (limit_neg, limit_pos) = self.limits();
        let file = File::create(filename)?;
        let writer = BufWriter::new(file);
        let mut encoder = Encoder::new(writer, 9)?;

        encoder.multithread(4)?;

        let mut encoder = encoder.auto_finish();

        encoder.write(&(self.time_limit as u64).to_le_bytes())?;

        encoder.write(&(self.num_directions as u64).to_le_bytes())?;

        for t in 0..=limit_pos as usize {
            for d in 0..self.num_directions as usize{
                for x in limit_neg..=limit_pos {
                    for y in limit_neg..=limit_pos {
                        encoder.write(&self.at(x, y, d, t).to_le_bytes())?;
                    }
                }
            }
        }

        for x in limit_neg..=limit_pos {
            for y in limit_neg..=limit_pos {
                encoder.write(&(self.field_type_at(x, y) as u64).to_le_bytes())?;
            }
        }

        Ok(())
    }
}

fn apply_kernel(
    table_old: &Vec<Vec<f64>>,
    kernels: &Vec<Kernel>,
    field_types: &Vec<Vec<usize>>,
    (limit_neg, limit_pos): (isize, isize),
    x: isize,
    y: isize,
) -> f64 {
    let field_type = field_types[(limit_pos + x) as usize][(limit_pos + y) as usize];
    let kernel = kernels[field_type].clone();

    let ks = (kernel.size() / 2) as isize;
    let mut sum = 0.0;

    for i in x - ks..=x + ks {
        if i < limit_neg || i > limit_pos {
            continue;
        }

        for j in y - ks..=y + ks {
            if j < limit_neg || j > limit_pos {
                continue;
            }

            // Kernel coordinates are inverted offset, i.e. -(i - x) and -(j - y)
            let kernel_x = x - i;
            let kernel_y = y - j;

            sum += table_old[(limit_pos + i) as usize][(limit_pos + j) as usize]
                * kernel.prob_at(kernel_x, kernel_y);
        }
    }

    sum
}

#[cfg(not(tarpaulin_include))]
impl Debug for CorDynamicProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicProgram")
            .field("time_limit", &self.time_limit)
            .finish()
    }
}

impl PartialEq for CorDynamicProgram {
    fn eq(&self, other: &Self) -> bool {
        self.time_limit == other.time_limit
            && self.table == other.table
            && self.field_types == other.field_types
    }
}

impl Eq for CorDynamicProgram {}

pub struct DynamicProgramLayerIterator {
    pub(crate) last_layer: Vec<Vec<Vec<f64>>>,
    pub(crate) layer: usize,
    pub(crate) dp: Option<CorDynamicProgram>,
    pub(crate) time_limit: usize,
    pub(crate) num_directions: usize,
    pub(crate) kernels: Vec<Kernel>,
    pub(crate) field_types: Vec<Vec<usize>>,
}

// impl Iterator for DynamicProgramLayerIterator {
//     type Item = Vec<Vec<f64>>;

//     fn next(&mut self) -> Option<Self::Item> {
//         // if self.layer >= self.time_limit {
//         //     return None;
//         // }

//         // if self.layer == 0 {
//         //     self.last_layer = vec![vec![vec![0.0; 2 * self.time_limit + 1]; 2 * self.time_limit + 1], n; num_directions];
//         //     self.last_layer[self.time_limit][self.time_limit] = 1.0;
//         //     self.layer += 1;

//         //     let mut table =
//         //         vec![vec![vec![0.0; 2 * self.time_limit + 1]; 2 * self.time_limit + 1]; 2];
//         //     table[0] = self.last_layer.clone();

//         //     self.dp = Some(CorDynamicProgram {
//         //         table,
//         //         time_limit: self.time_limit,
//         //         num_directions: self.num_directions,
//         //         kernels: self.kernels.clone(),
//         //         field_types: self.field_types.clone(),
//         //     });

//         //     return Some(self.last_layer.clone());
//         // }

//         // let mut table = vec![vec![vec![0.0; 2 * self.time_limit + 1]; 2 * self.time_limit + 1]; 2];
//         // table[0] = self.last_layer.clone();

//         // let dp = self.dp.as_mut().unwrap();
//         // dp.table = table;

//         // let (limit_neg, limit_pos) = dp.limits();
        
//         // for x in limit_neg..=limit_pos {
//         //     for y in limit_neg..=limit_pos {
//         //         // Revisit
//         //         dp.apply_kernel_at(x, y, d, 1);
//         //     }
//         // }

//         // self.last_layer = dp.table[1].clone();
//         // self.layer += 1;

//         // Some(dp.table[1][0].clone())
//     }
// }

pub fn compute_multiple(dps: &mut [CorDynamicProgram]) {
    dps.par_iter_mut().for_each(|dp| dp.compute());
}

pub fn compute_multiple_save(dps: Vec<CorDynamicProgram>, filename: String) {
    let dps = dps.into_iter().zip((0..).into_iter()).collect::<Vec<_>>();

    dps.into_par_iter().for_each(|(mut dp, i)| {
        dp.compute();
        dp.save(format!("{filename}_{i}.zst")).unwrap();
    });
}

// pub fn compute_multiple_save_layered(dps: Vec<CorDynamicProgram>, path: String) {
//     let dps = dps.into_iter().zip((0..).into_iter()).collect::<Vec<_>>();

//     dps.into_par_iter().for_each(|(mut dp, i)| {
//         debug!("Computing dp {i}");
//         dp.compute();

//         let (limit_neg, limit_pos) = dp.limits();

//         debug!("Saving dp {i}");
//         for t in 0..=limit_pos as usize {
//             if !Path::new(&path).join(format!("{i}")).exists() {
//                 fs::create_dir(Path::new(&path).join(format!("{i}")))
//                     .expect("Could not create directory");
//             }

//             let path = Path::new(&path)
//                 .join(format!("{i}"))
//                 .join(format!("{t}.dp"));
//             let file = File::create(&path).expect("Could not create file");
//             let mut writer = BufWriter::new(file);

//             for x in limit_neg..=limit_pos {
//                 for y in limit_neg..=limit_pos {
//                     writer
//                         .write(&dp.at(x, y, d, t).to_le_bytes())
//                         .expect("Could not write to file");
//                 }
//             }
//         }
//     });
// }

#[cfg(test)]
mod tests {
    use crate::dp::builder::DynamicProgramBuilder;
    use crate::dp::{DynamicProgramPool, DynamicPrograms};
    use crate::kernel::biased_rw::BiasedRwGenerator;
    use crate::kernel::simple_rw::SimpleRwGenerator;
    use crate::kernel::{Direction, Kernel};

    #[test]
    fn test_simple_dp_at() {
        let mut dp = DynamicProgramBuilder::new()
            .simple()
            .time_limit(10)
            .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
            .build()
            .unwrap();

        dp.compute();

        let DynamicProgramPool::Single(dp) = dp else {
            unreachable!();
        };

        assert_eq!(dp.at(0, 0, 0), 1.0);
    }

    #[test]
    fn test_simple_dp_set() {
        let dp = DynamicProgramBuilder::new()
            .simple()
            .time_limit(10)
            .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
            .build()
            .unwrap();

        let DynamicProgramPool::Single(mut dp) = dp else {
            unreachable!();
        };

        dp.set(0, 0, 0, 10.0);

        assert_eq!(dp.at(0, 0, 0,), 10.0);
    }

    #[test]
    fn test_compute() {
        let mut dp = DynamicProgramBuilder::new()
            .simple()
            .time_limit(100)
            .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
            .build()
            .unwrap();

        dp.compute();

        let DynamicProgramPool::Single(dp) = dp else {
            unreachable!();
        };

        assert_eq!(dp.at(0, 0, 1), 0.2);
        assert_eq!(dp.at(-1, 0, 1), 0.2);
        assert_eq!(dp.at(1, 0, 1), 0.2);
        assert_eq!(dp.at(0, -1, 1), 0.2);
        assert_eq!(dp.at(0, 1, 1), 0.2);
    }

    #[test]
    fn test_dp_eq() {
        let mut dp1 = DynamicProgramBuilder::new()
            .simple()
            .time_limit(10)
            .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
            .build()
            .unwrap();

        dp1.compute();

        let mut dp2 = DynamicProgramBuilder::new()
            .simple()
            .time_limit(10)
            .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
            .build()
            .unwrap();

        dp2.compute();

        let DynamicProgramPool::Single(dp1) = dp1 else {
            unreachable!();
        };
        let DynamicProgramPool::Single(dp2) = dp2 else {
            unreachable!();
        };

        assert_eq!(dp1, dp2);
    }

    #[test]
    fn test_dp_not_eq() {
        let mut dp1 = DynamicProgramBuilder::new()
            .simple()
            .time_limit(10)
            .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
            .build()
            .unwrap();

        dp1.compute();

        let mut dp2 = DynamicProgramBuilder::new()
            .simple()
            .time_limit(10)
            .kernel(
                Kernel::from_generator(BiasedRwGenerator {
                    probability: 0.5,
                    direction: Direction::North,
                })
                .unwrap(),
            )
            .build()
            .unwrap();

        dp2.compute();

        let DynamicProgramPool::Single(dp1) = dp1 else {
            unreachable!();
        };
        let DynamicProgramPool::Single(dp2) = dp2 else {
            unreachable!();
        };

        assert_ne!(dp1, dp2);
    }
}
