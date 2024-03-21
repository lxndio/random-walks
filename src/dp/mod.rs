//! Provides the dynamic programs required to compute random walks.
//!
//! This library contains different dynamic programs which must be computed using some specified
//! kernel. After the computation, random walks can be generated using the tables of the dynamic
//! program.
//!
//! # Types
//!
//! There are two different types of dynamic programs which compute the random walk probabilities.
//! They are listed below together with short descriptions.
//!
//! - [`DynamicProgram`]: A dynamic program that uses a single kernel to compute the
//! probabilities.
//!
//! Dynamic programs are wrapped into the [`DynamicProgramPool`] enum and must
//! implement the [`DynamicPrograms`] trait.
//!
//! # Examples
//!
//! ## Creating a Dynamic Program
//!
//! Dynamic programs can be created using the
//! [`DynamicProgramBuilder`](builder::DynamicProgramBuilder). It offers different options for
//! dynamic programs which are described in detail in the [`builder`] module. The general structure,
//! however, looks like this:
//!
//! ```
//! use randomwalks_lib::dp::builder::DynamicProgramBuilder;
//! use randomwalks_lib::kernel::Kernel;
//! use randomwalks_lib::kernel::simple_rw::SimpleRwGenerator;
//!
//! let dp = DynamicProgramBuilder::new()
//!     .simple()
//!     .time_limit(400)
//!     .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
//!     .build()
//!     .unwrap();
//! ```
//!
//! In this example, a [`DynamicProgram`] is created with a time limit of 400 time steps.
//! As can be seen, a [`Kernel`](crate::kernel::Kernel) must be specified. More information on
//! kernels can be found in the documentation of the [`kernel`](crate::kernel) module.
//!
//! ## Computation
//!
//! After creation, a dynamic program is initialized but the actual values are not yet computed.
//! To do the computation,
//!
//! ```
//! # use randomwalks_lib::dp::builder::DynamicProgramBuilder;
//! # use randomwalks_lib::dp::DynamicPrograms;
//! # use randomwalks_lib::kernel::Kernel;
//! # use randomwalks_lib::kernel::simple_rw::SimpleRwGenerator;
//! #
//! # let mut dp = DynamicProgramBuilder::new()
//! #     .simple()
//! #     .time_limit(400)
//! #     .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
//! #     .build()
//! #     .unwrap();
//! #
//! dp.compute();
//! ```
//!
//! can be run.
//!

use std::{
    borrow::Borrow,
    fs::{self, File},
    io::{BufReader, Read},
    ops::Index,
    path::Path,
};

use glob::glob;
use log::{debug, trace};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zstd::Decoder;

use crate::dp::simple::DynamicProgram;

use self::simple::DynamicProgramLayerIterator;

pub mod builder;
pub mod simple;

pub trait DynamicPrograms {
    fn limits(&self) -> (isize, isize);

    fn compute(&mut self);

    fn compute_parallel(&mut self);

    fn field_types(&self) -> Vec<Vec<usize>>;

    #[cfg(feature = "plotting")]
    fn heatmap(&self, path: String, t: usize) -> anyhow::Result<()>;

    fn print(&self, t: usize);

    fn save(&self, filename: String) -> std::io::Result<()>;
}

#[derive(Error, Debug)]
pub enum DynamicProgramError {
    /// This error occurs when try_unwrap() is called on a `DynamicProgramPool` holding multiple
    /// dynamic programs.
    #[error("try_unwrap() can only be called on a single dynamic program")]
    UnwrapOnMultiple,

    /// This error occurs when try_into_iter() is called on a `DynamicProgramPool` holding multiple
    /// dynamic programs.
    #[error("try_into_iter() can only be called on a single dynamic program")]
    IntoIterOnMultiple,
}

pub struct DynamicProgramDiskVec {
    path: String,
    len: usize,
    time_limit: usize,
}

impl DynamicProgramDiskVec {
    pub fn try_new(path: String) -> std::io::Result<Self> {
        let file = File::open(Path::new(&path).join("layer_0.zst"))?;
        let reader = BufReader::new(file);
        let mut decoder = Decoder::new(reader)?;

        let len = glob(Path::new(&path).join("layer_*.zst").to_str().unwrap())
            .unwrap()
            .count();

        let mut time_limit = [0u8; 8];
        decoder.read_exact(&mut time_limit)?;
        let time_limit = u64::from_le_bytes(time_limit) as usize;

        debug!("Initializing dynamic program disk vector with {len} elements and a time limit of {time_limit} time steps");

        Ok(Self {
            path,
            len,
            time_limit,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn time_limit(&self) -> usize {
        self.time_limit
    }

    pub fn try_at(&self, x: isize, y: isize, t: usize, variant: usize) -> Option<f64> {
        if t >= self.time_limit {
            debug!("Time step {t} out of bounds");
            return None;
        }

        if variant >= self.len {
            debug!("Variant {variant} out of bounds");
            return None;
        }

        trace!("Reading value at ({x}, {y}) at time step {t} for variant {variant} from disk");

        let file = File::open(Path::new(&self.path).join(format!("layer_{t}.zst"))).ok()?;
        let reader = BufReader::new(file);
        let mut decoder = Decoder::new(reader).ok()?;

        let mut header = [0u8; 16];
        decoder.read_exact(&mut header).ok()?;

        // Skip variants until the correct one is reached
        for i in 0..variant {
            let mut buf = [0u8; 8];
            for _ in 0..4 * self.time_limit + 2 {
                decoder.read_exact(&mut buf).ok()?;
            }
        }

        // Read correct variant's layer
        let mut layer = vec![vec![0.0; 2 * self.time_limit + 1]; 2 * self.time_limit + 1];
        let mut buf = [0u8; 8];

        for x in 0..2 * self.time_limit + 1 {
            for y in 0..2 * self.time_limit + 1 {
                decoder.read_exact(&mut buf).ok()?;
                layer[x][y] = f64::from_le_bytes(buf);
            }
        }

        Some(
            layer[(self.time_limit as isize + x) as usize][(self.time_limit as isize + y) as usize],
        )
    }

    pub fn at(&self, x: isize, y: isize, t: usize, variant: usize) -> f64 {
        match self.try_at(x, y, t, variant) {
            Some(value) => value,
            None => panic!("Could not read value from dynamic program disk vector"),
        }
    }

    pub fn at_or(&self, x: isize, y: isize, t: usize, variant: usize, default: f64) -> f64 {
        match self.try_at(x, y, t, variant) {
            Some(value) => value,
            None => default,
        }
    }
}

// let (limit_neg, limit_pos) = dp.limits();
// let mut buf = [0u8; 8];

// for t in 0..=limit_pos as usize {
//     for x in limit_neg..=limit_pos {
//         for y in limit_neg..=limit_pos {
//             decoder.read_exact(&mut buf)?;
//             dp.set(x, y, t, f64::from_le_bytes(buf));
//         }
//     }
// }

pub enum DynamicProgramPool {
    Single(DynamicProgram),
    Multiple(Vec<DynamicProgram>),
    MultipleFromDisk(DynamicProgramDiskVec),
}

#[cfg(not(tarpaulin_include))]
impl DynamicProgramPool {
    pub fn try_unwrap(&self) -> Result<&DynamicProgram, DynamicProgramError> {
        match self {
            DynamicProgramPool::Single(single) => Ok(single),
            _ => Err(DynamicProgramError::UnwrapOnMultiple),
        }
    }

    pub fn try_unwrap_mut(&mut self) -> Result<&mut DynamicProgram, DynamicProgramError> {
        match self {
            DynamicProgramPool::Single(single) => Ok(single),
            _ => Err(DynamicProgramError::UnwrapOnMultiple),
        }
    }

    pub fn try_into(self) -> Result<DynamicProgram, DynamicProgramError> {
        match self {
            DynamicProgramPool::Single(single) => Ok(single),
            _ => Err(DynamicProgramError::UnwrapOnMultiple),
        }
    }

    pub fn at(
        &self,
        x: isize,
        y: isize,
        t: usize,
        variant: usize,
    ) -> Result<f64, DynamicProgramError> {
        match self {
            DynamicProgramPool::Single(single) => Ok(single.at(x, y, t)),
            DynamicProgramPool::Multiple(multiple) => Ok(multiple[variant].at(x, y, t)),
            DynamicProgramPool::MultipleFromDisk(disk_vec) => Ok(disk_vec.at(x, y, t, variant)),
        }
    }

    pub fn at_or(
        &self,
        x: isize,
        y: isize,
        t: usize,
        variant: usize,
        default: f64,
    ) -> Result<f64, DynamicProgramError> {
        match self {
            DynamicProgramPool::Single(single) => Ok(single.at_or(x, y, t, default)),
            DynamicProgramPool::Multiple(multiple) => Ok(multiple[variant].at_or(x, y, t, default)),
            DynamicProgramPool::MultipleFromDisk(disk_vec) => {
                Ok(disk_vec.at_or(x, y, t, variant, default))
            }
        }
    }
}

#[cfg(not(tarpaulin_include))]
impl DynamicPrograms for DynamicProgramPool {
    /// Wrapper for `SimpleDynamicProgram::limits()`. Fails if called on a `DynamicProgramPool`
    /// holding multiple dynamic programs.
    fn limits(&self) -> (isize, isize) {
        self.try_unwrap().unwrap().limits()
    }

    /// Wrapper for `SimpleDynamicProgram::compute()`. Fails if called on a `DynamicProgramPool`
    /// holding multiple dynamic programs.
    fn compute(&mut self) {
        self.try_unwrap_mut().unwrap().compute()
    }

    /// Wrapper for `SimpleDynamicProgram::compute_parallel()`. Fails if called on a
    /// `DynamicProgramPool` holding multiple dynamic programs.
    fn compute_parallel(&mut self) {
        self.try_unwrap_mut().unwrap().compute_parallel()
    }

    /// Wrapper for `SimpleDynamicProgram::field_types()`. Fails if called on a `DynamicProgramPool`
    /// holding multiple dynamic programs.
    fn field_types(&self) -> Vec<Vec<usize>> {
        self.try_unwrap().unwrap().field_types()
    }

    /// Wrapper for `SimpleDynamicProgram::heatmap()`. Fails if called on a `DynamicProgramPool`
    /// holding multiple dynamic programs.
    #[cfg(feature = "plotting")]
    fn heatmap(&self, path: String, t: usize) -> anyhow::Result<()> {
        self.try_unwrap().unwrap().heatmap(path, t)
    }

    /// Wrapper for `SimpleDynamicProgram::print()`. Fails if called on a `DynamicProgramPool`
    /// holding multiple dynamic programs.
    fn print(&self, t: usize) {
        self.try_unwrap().unwrap().print(t)
    }

    /// Wrapper for `SimpleDynamicProgram::save()`. Fails if called on a `DynamicProgramPool`
    /// holding multiple dynamic programs.
    fn save(&self, filename: String) -> std::io::Result<()> {
        self.try_unwrap().unwrap().save(filename)
    }
}

#[derive(Default, Clone, PartialEq, Serialize, Deserialize, Debug)]
pub enum DynamicProgramType {
    #[default]
    Simple,
}
