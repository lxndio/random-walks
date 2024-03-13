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

use std::{borrow::Borrow, fs, ops::Index};

use log::trace;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::dp::simple::DynamicProgram;

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
}

pub struct DynamicProgramDiskVec {
    path: String,
    len: usize,
}

impl DynamicProgramDiskVec {
    pub fn try_new(path: String) -> std::io::Result<Self> {
        let len = fs::read_dir(&path)?.count();
        trace!("Initializing dynamic program disk vector with {len} elements");

        Ok(Self { path, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, index: usize) -> Option<DynamicProgram> {
        if index >= self.len {
            return None;
        }

        let path = format!("{}/dp_{}.zst", self.path, index);
        trace!("Loading dynamic program from {path}");

        Some(
            DynamicProgram::load(path)
                .expect("could not load dynamic program")
                .try_into()
                .unwrap(),
        )
    }
}

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
