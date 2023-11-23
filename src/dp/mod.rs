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
//! - [`SimpleDynamicProgram`]: A dynamic program that uses a single kernel to compute the
//! probabilities.
//! - [`MultiDynamicProgram`]: A dynamic program that uses multiple kernels to compute the
//! probabilities. This is for example required when using correlated random walks.
//!
//! Dynamic programs are wrapped into the [`DynamicProgram`] enum and must
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
//! In this example, a [`SimpleDynamicProgram`] is created with a time limit of 400 time steps.
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

use crate::dp::multi::MultiDynamicProgram;
use crate::dp::simple::SimpleDynamicProgram;
use serde::{Deserialize, Serialize};

pub mod builder;
pub mod multi;
pub mod simple;

pub trait DynamicPrograms {
    fn limits(&self) -> (isize, isize);

    fn compute(&mut self);

    fn compute_parallel(&mut self);

    fn field_types(&self) -> Vec<Vec<usize>>;

    #[cfg(feature = "plotting")]
    fn heatmap(&self, path: String, t: usize) -> anyhow::Result<()>;

    fn print(&self, t: usize);

    fn save(&self, filename: String) -> anyhow::Result<()>;
}

pub enum DynamicProgram {
    Simple(SimpleDynamicProgram),
    Multi(MultiDynamicProgram),
}

#[cfg(not(tarpaulin_include))]
impl DynamicProgram {
    fn unwrap(&self) -> &dyn DynamicPrograms {
        match self {
            DynamicProgram::Simple(simple) => simple,
            DynamicProgram::Multi(multi) => multi,
        }
    }

    fn unwrap_mut(&mut self) -> &mut dyn DynamicPrograms {
        match self {
            DynamicProgram::Simple(simple) => simple,
            DynamicProgram::Multi(multi) => multi,
        }
    }
}

#[cfg(not(tarpaulin_include))]
impl DynamicPrograms for DynamicProgram {
    fn limits(&self) -> (isize, isize) {
        self.unwrap().limits()
    }

    fn compute(&mut self) {
        self.unwrap_mut().compute()
    }

    fn compute_parallel(&mut self) {
        self.unwrap_mut().compute_parallel()
    }

    fn field_types(&self) -> Vec<Vec<usize>> {
        self.unwrap().field_types()
    }

    #[cfg(feature = "plotting")]
    fn heatmap(&self, path: String, t: usize) -> anyhow::Result<()> {
        self.unwrap().heatmap(path, t)
    }

    fn print(&self, t: usize) {
        self.unwrap().print(t)
    }

    fn save(&self, filename: String) -> anyhow::Result<()> {
        self.unwrap().save(filename)
    }
}

#[derive(Default, Clone, PartialEq, Serialize, Deserialize, Debug)]
pub enum DynamicProgramType {
    #[default]
    Simple,
    Multi,
}
