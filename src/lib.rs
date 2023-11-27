//! A library for efficient movement interpolation using different random walk models.
//! Below is an overview of all important types and functions, a short guide on how to get started
//! using the library, as well as some examples.
//!
//! # Walk Models
//! This library implements different random walk models which can be used to generate
//! random walks with different characteristics. Below is a list of the different models
//! including short descriptions.
//!
//! - [Simple random walk](kernel::simple_rw::SimpleRwGenerator): Generates completely random walks
//! - [Biased random walk](kernel::biased_rw::BiasedRwGenerator): Generates random walks which tend
//! into a specific direction, i.e. they have a bias
//! - [Correlated random walk](kernel::correlated_rw::CorrelatedRwGenerator): Generates random walks
//! which have a higher probability of taking the same direction as in the last step
//! - [Biased and correlated random walk](kernel::biased_correlated_rw::BiasedCorrelatedRwGenerator)
//! : Combines the biased random walk with the correlated random walk
//! - [Lévy walk](kernel::levy_walk::LevyWalkGenerator): Generates random walks which sometimes jump
//! a few steps at once
//!
//! Walk models are implemented in this library as so-called
//! [`KernelGenerator`](kernel::generator::KernelGenerator)s. What they do is essentially to
//! generate a probability [`Kernel`](kernel::Kernel) that models the corresponding random walk
//! model. Alternatively, kernels can be creates by hand using some custom probability distribution.
//! This can be done the easiest using the [`kernel!`] macro.
//!
//! # Dynamic Programs
//!
//! There are two different types of dynamic programs which compute the random walk probabilities.
//! They are listed below together with short descriptions.
//!
//! - [`SimpleDynamicProgram`](dp::simple::DynamicProgram): A dynamic program that uses a
//! single kernel to compute the probabilities.
//! - [`MultiDynamicProgram`](dp::multi::MultiDynamicProgram): A dynamic program that uses multiple
//! kernels to compute the probabilities. This is for example required when using correlated
//! random walks.
//!
//! Dynamic programs are wrapped into the [`DynamicProgram`](dp::DynamicProgramPool) enum and must
//! implement the [`DynamicPrograms`](dp::DynamicPrograms) trait. They can be initialized using the
//! [`DynamicProgramBuilder`](dp::builder::DynamicProgramBuilder).
//!
//! # Walkers
//!
//! Walkers generate random walks on the basis of a previously computed dynamic program. There
//! are three different walkers available which do slightly different things.
//!
//! - [`StandardWalker`](walker::standard::StandardWalker): The standard walker for generating
//! random walks that works with all kernels using the `SimpleDynamicProgram`.
//! - [`CorrelatedWalker`](walker::correlated::CorrelatedWalker): A special walker that is designed
//! to work with the `MultiDynamicProgram` using kernels for correlated random walks. In each step,
//! it chooses a different dynamic program table depending on the direction of the last step.
//! - [`MultiStepWalker`](walker::multi_step::MultiStepWalker): Like the `StandardWalker` but it
//! allows multiple steps to be made at once, making use of dynamic programs that were generated
//! with kernels larger than 3x3.
//!
//! # Dataset Functionality
//!
//! [`Dataset`s](dataset::Dataset) allow automatic generation of random walks based on many
//! location points. The easiest way to load or generate a dataset is using the
//! [`DatasetBuilder`](dataset::builder::DatasetBuilder). Using the builder, a dataset can be
//! generated or loaded from a CSV file or a Polars `DataFrame` (if `polars` feature is enabled).
//!
//! See the documentation of `Dataset` for more information on how to work with datasets. There
//! are different functions to modify, e.g. filter, datasets. Single random walks can be generated
//! using the function [`rw_between()`](dataset::Dataset::rw_between). To generate many random
//! walks at once, use the [`DatasetWalksBuilder`](dataset::DatasetWalksBuilder).
//!
//! # Features
//!
//! This library has the following features which enable additional functionality.
//!
//! - `plotting`: Allows generating plots of random walks and datasets and save them as images.
//! - `polars_loading`: Allows loading `DataFrame`s from the
//! [Polars](https://crates.io/crates/polars) crate.
//!
//! # Getting Started
//!
//! The normal workflow when using this library may be as follows:
//!
//! 1. Building and computing a dynamic program
//! 2. Loading a dataset
//! 3. Processing the dataset
//! 4. Generating random walks between points of the dataset
//!
//! Step 1 can be accomplished by using the
//! [`DynamicProgramBuilder`](dp::builder::DynamicProgramBuilder). Using that, a kernel for a
//! specific walk model can also be specified. For step 2, the
//! [`DatasetBuilder`](dataset::builder::DatasetBuilder) may be used. Step 3 may consist of
//! converting the coordinates in the dataset from GCS to XY coordinates, filtering for specific
//! entries and shrinking the dataset for better workability. See the documentation on
//! [`Dataset`s](dataset::Dataset) for further information on how to perform these tasks. For
//! step 4, the [`DatasetWalksBuilder`](dataset::DatasetWalksBuilder) can be used.
//!
//! # Examples
//!
//! These examples should give a brief overview of the framework's functionality and give the user
//! a starting point to working with it. More detailed examples can be found on the documentation
//! pages of different features.
//!
//! ## Generating a random walk
//!
//! This example shows how to build and compute a dynamic program, and how to use it to generate
//! a random walk from the origin `(0, 0)` to `(100, 50)` in 400 time steps.
//!
//! ```
//! use randomwalks_lib::dp::builder::DynamicProgramBuilder;
//! use randomwalks_lib::dp::DynamicPrograms;
//! use randomwalks_lib::kernel::Kernel;
//! use randomwalks_lib::kernel::simple_rw::SimpleRwGenerator;
//! use randomwalks_lib::walker::standard::StandardWalker;
//! use randomwalks_lib::walker::Walker;
//!
//! let mut dp = DynamicProgramBuilder::new()
//!     .simple()
//!     .time_limit(400)
//!     .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
//!     .build()
//!     .unwrap();
//!
//! dp.compute();
//!
//! let walker = StandardWalker;
//! let walk = walker.generate_path(&dp, 100, 50, 400).unwrap();
//! ```
//!
//! ## Loading and using a dataset
//!
//! This example shows how to load a dataset, do some simple preprocessing, and compute a random
//! walk between its first and second datapoint in 400 time steps. Assume that `dp` is a dynamic
//! program that has already been computed, e.g. as seen in the example above.
//!
//! ```
//! use randomwalks_lib::dataset::builder::DatasetBuilder;
//! use randomwalks_lib::dataset::loader::{ColumnAction, CoordinateType};
//! use randomwalks_lib::walker::standard::StandardWalker;
//!
//! let mut dataset = DatasetBuilder::new()
//!     .from_csv("dataset.csv")
//!     .add_column_actions(vec![
//!         ColumnAction::KeepX,
//!         ColumnAction::KeepY,
//!         ColumnAction::KeepMetadata("agent_id")
//!     ])
//!     .coordinate_type(CoordinateType::XY)
//!     .build()
//!     .unwrap();
//!
//! let walker = StandardWalker;
//! let walk = dataset.rw_between(&dp, Box::new(walker), 0, 1, 400);
//! ```
//!

pub mod dataset;
pub mod dp;
pub mod kernel;
pub mod walk;
pub mod walker;
pub mod vec2d;
