//! Provides a builder for dynamic programs.
//!
//! The [`DynamicProgramBuilder`] is used to create and initialize new
//! [`DynamicProgram`s](crate::dp::DynamicProgramPool). In the following, a short overview of all
//! options will be given.
//!
//! # Required Options
//!
//! A typical usage of the [`DynamicProgramBuilder`] could look like this:
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
//!     .build();
//! ```
//!
//! In this example, a [`DynamicProgram`] is created with a time limit of 400 time steps.
//! As can be seen, a [`Kernel`](crate::kernel::Kernel) must be specified. More information on
//! kernels can be found in the documentation of the [`kernel`](crate::kernel) module.
//!
//! Alternatively, a [`MultiDynamicProgram`] can be created using the
//! [`multi()`](DynamicProgramBuilder::multi) function. When using this, instead of a single kernel,
//! multiple kernels have to be specified using the [`kernels()`](DynamicProgramBuilder::kernels)
//! function.
//!
//! After calling [`build()`](DynamicProgramBuilder::build), the builder will return either a
//! [`DynamicProgram`](crate::dp::DynamicProgramPool) or a
//! [`DynamicProgramBuilderError`](DynamicProgramBuilderError).
//!
//! # Barriers & Field Probabilities
//!
//! If desired, barriers can be added to the map. These can either be completely blocking or reduce
//! the possibility of walks going through them by a specific amount. They can be added as follows.
//!
//! ```
//! # use randomwalks_lib::dp::builder::DynamicProgramBuilder;
//! # use randomwalks_lib::kernel::Kernel;
//! # use randomwalks_lib::kernel::simple_rw::SimpleRwGenerator;
//! # use randomwalks_lib::xy;
//! #
//! let dp = DynamicProgramBuilder::new()
//!     // ...
//!     .add_single_barrier(xy!(10, 10))
//!     .add_rect_barrier(xy!(10, -10), xy!(10, 10))
//!     // ...
//!     .build();
//! ```
//! [`add_single_barrier()`](DynamicProgramBuilder::add_single_barrier) block just a single field on
//! the map by reducing its field probability to 0. Therefore, no walk with use that field anymore.
//!
//! Using [`add_rect_barrier()`](DynamicProgramBuilder::add_rect_barrier), all fields in the given
//! range (in the example `[10, -10]` to `[10, 10]`) are blocked for walks to use.
//!
//! If other forms of barriers are required or if some fields should not be entirely blocked but the
//! usage probability should be reduced,
//! [`field_probabilities()`](DynamicProgramBuilder::field_probabilities) can be used. This function
//! allows to set the probability of each field separately. A probability of `0.0` means that the
//! field is not visited in any way, while a probability of `1.0` means that the field has its
//! normal probability that was assigned to it while computing the dynamic program.

use crate::dataset::point::XYPoint;
use crate::dp::simple::DynamicProgram;
use crate::dp::{DynamicProgramPool, DynamicProgramType};
use crate::kernel::Kernel;
use num::Zero;
use std::collections::HashMap;
use thiserror::Error;

/// An error that can occur when using a [`DynamicProgramBuilder`].
#[derive(Error, Debug)]
pub enum DynamicProgramBuilderError {
    /// This error occurs when no type of dynamic program was specified using
    /// [`simple()`](DynamicProgramBuilder::simple) or
    /// [`multi()`](DynamicProgramBuilder::multi).
    #[error("a type of dynamic program must be chosen")]
    NoTypeSet,

    /// This error occurs when no time limit was set using
    /// [`time_limit()`](DynamicProgramBuilder::time_limit).
    #[error("a time limit must be set")]
    NoTimeLimitSet,

    /// This error occurs when no kernels were set using
    /// [`kernels()`](DynamicProgramBuilder::kernels).
    #[error("kernels must be set")]
    NoKernelsSet,

    /// This error occurs when [`multi()`](DynamicProgramBuilder::multi) was used, but only
    /// a single kernel was given using [`kernel()`](DynamicProgramBuilder::kernel). Use
    /// [`kernels()`](DynamicProgramBuilder::kernels) instead.
    #[error("a multi DP takes multiple kernels and not a single one")]
    SingleKernelForMulti,

    /// This error occurs when [`single()`](DynamicProgramBuilder::single) was used, but multiple
    /// kernels were given using [`kernels()`](DynamicProgramBuilder::kernels). Use
    /// [`kernel()`](DynamicProgramBuilder::kernel) instead.
    #[error("a simple DP takes one kernel and not multiple ones")]
    MultipleKernelsForSimple,

    /// This error occurs when the size of the vector of field probabilities given using
    /// [`field_probabilities()`](DynamicProgramBuilder::field_probabilities) does not match
    /// the size of the dynamic program's table.
    #[error("field probabilities must be of same size as DP table")]
    WrongSizeOfFieldProbabilities,

    /// This error occurs when a barrier that was given using
    /// [`add_single_barrier()`](DynamicProgramBuilder::add_single_barrier) or
    /// [`add_rect_barrier()`](DynamicProgramBuilder::add_rect_barrier) is entirely or partially
    /// out of range of the dynamic program's table.
    #[error("barriers must be inside the time limit range")]
    BarrierOutOfRange,
}

/// A builder used to create and initialize dynamic programs.
///
/// For a detailed description and examples see the documentation of the
/// [`builder`](crate::dp::builder) module.
#[derive(Default)]
pub struct DynamicProgramBuilder {
    time_limit: Option<usize>,
    dp_type: Option<DynamicProgramType>,
    kernels: Option<Vec<(usize, Kernel)>>,
    field_probabilities: Option<Vec<Vec<f64>>>,
    field_types: Option<Vec<Vec<usize>>>,
    barriers: Vec<XYPoint>,
}

impl DynamicProgramBuilder {
    /// Creates a new [`DynamicProgramBuilder`].
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    /// Sets the type of the dynamic program as a
    /// [`DynamicProgram`].
    pub fn simple(mut self) -> Self {
        self.dp_type = Some(DynamicProgramType::Simple);

        self
    }

    /// Sets the type of the dynamic program as a
    /// [`MultiDynamicProgram`].
    pub fn multi(mut self) -> Self {
        todo!();
    }

    /// Sets the type of the dynamic program to the specified
    /// [`DynamicProgramType`].
    pub fn with_type(mut self, dp_type: DynamicProgramType) -> Self {
        self.dp_type = Some(dp_type);

        self
    }

    /// Sets the time limit for the dynamic program.
    pub fn time_limit(mut self, time_limit: usize) -> Self {
        self.time_limit = Some(time_limit);

        self
    }

    /// Sets the [`Kernel`](crate::kernel::Kernel) for the dynamic program. Use this in combination
    /// with a [`DynamicProgram`].
    pub fn kernel(mut self, kernel: Kernel) -> Self {
        self.kernels(vec![(0, kernel)])
    }

    pub fn kernels(mut self, kernels: Vec<(usize, Kernel)>) -> Self {
        self.kernels = Some(kernels);

        self
    }

    pub fn field_types(mut self, types: Vec<Vec<usize>>) -> Self {
        self.field_types = Some(types);

        self
    }

    /// Adds a single barrier to the dynamic program.
    pub fn add_single_barrier(mut self, at: XYPoint) -> Self {
        self.barriers.push(at);

        self
    }

    /// Adds multiple barriers in a specified rectangular area to the dynamic program.
    pub fn add_rect_barrier(mut self, from: XYPoint, to: XYPoint) -> Self {
        for x in from.x..=to.x {
            for y in from.y..=to.y {
                self.barriers.push(XYPoint { x, y })
            }
        }

        self
    }

    /// Builds the dynamic program.
    ///
    /// This builds the dynamic program after all options have been specified. Returns a
    /// [`DynamicProgramPool`] if successful.
    ///
    /// # Errors
    ///
    /// Returns a [`DynamicProgramBuilderError`] if misconfigured.
    pub fn build(self) -> Result<DynamicProgramPool, DynamicProgramBuilderError> {
        let Some(time_limit) = self.time_limit else {
            return Err(DynamicProgramBuilderError::NoTimeLimitSet);
        };
        let Some(dp_type) = self.dp_type else {
            return Err(DynamicProgramBuilderError::NoTypeSet);
        };

        let mut field_probabilities = match self.field_probabilities {
            Some(fp) => {
                if fp.len() != 2 * time_limit + 1 {
                    return Err(DynamicProgramBuilderError::WrongSizeOfFieldProbabilities);
                }

                for fpp in fp.iter() {
                    if fpp.len() != 2 * time_limit + 1 {
                        return Err(DynamicProgramBuilderError::WrongSizeOfFieldProbabilities);
                    }
                }

                fp
            }
            None => vec![vec![1.0; 2 * time_limit + 1]; 2 * time_limit + 1],
        };

        let mut field_types = match self.field_types {
            Some(ft) => ft,
            None => vec![vec![0; 2 * time_limit + 1]; 2 * time_limit + 1],
        };

        for (x, y) in self.barriers.iter().map(|p| <(i64, i64)>::from(*p)) {
            if x < -(time_limit as i64)
                || x > time_limit as i64
                || y < -(time_limit as i64)
                || y > time_limit as i64
            {
                return Err(DynamicProgramBuilderError::BarrierOutOfRange);
            }

            let x = (time_limit as i64 + x) as usize;
            let y = (time_limit as i64 + y) as usize;

            field_probabilities[x][y] = 0.0;
        }

        match dp_type {
            DynamicProgramType::Simple => {
                let Some(mut kernels) = self.kernels else {
                    return Err(DynamicProgramBuilderError::NoKernelsSet);
                };

                // Map field types to contiguous value range

                let mut kernels_mapped = Vec::new();
                let mut field_type_map = HashMap::new();
                let mut i = 0usize;

                for (field_type, kernel) in kernels.iter() {
                    kernels_mapped.push(kernel.clone());
                    field_type_map.insert(field_type, i);
                    i += 1;
                }

                for x in 0..2 * time_limit + 1 {
                    for y in 0..2 * time_limit + 1 {
                        field_types[x][y] = field_type_map[&field_types[x][y]];
                    }
                }

                Ok(DynamicProgramPool::Single(DynamicProgram {
                    table: vec![
                        vec![vec![Zero::zero(); 2 * time_limit + 1]; 2 * time_limit + 1];
                        time_limit + 1
                    ],
                    time_limit,
                    kernels: kernels_mapped,
                    field_types,
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::point::XYPoint;
    use crate::dp::builder::{DynamicProgramBuilder, DynamicProgramBuilderError};
    use crate::dp::DynamicProgramType;
    use crate::kernel::correlated_rw::CorrelatedRwGenerator;
    use crate::kernel::simple_rw::SimpleRwGenerator;
    use crate::kernel::Kernel;
    use crate::xy;

    #[test]
    fn test_builder_missing_time_limit() {
        let dp = DynamicProgramBuilder::new().simple().build();

        assert!(matches!(
            dp,
            Err(DynamicProgramBuilderError::NoTimeLimitSet)
        ));
    }

    #[test]
    fn test_builder_missing_type() {
        let dp = DynamicProgramBuilder::new().time_limit(10).build();

        assert!(matches!(dp, Err(DynamicProgramBuilderError::NoTypeSet)));
    }

    // #[test]
    // fn test_wrong_size_of_field_probabilities() {
    //     let fps = vec![vec![1.0; 21]; 12];
    //
    //     let dp = DynamicProgramBuilder::new()
    //         .simple()
    //         .time_limit(10)
    //         .field_probabilities(fps)
    //         .build();
    //
    //     assert!(matches!(
    //         dp,
    //         Err(DynamicProgramBuilderError::WrongSizeOfFieldProbabilities)
    //     ));
    //
    //     let fps = vec![vec![1.0; 8]; 21];
    //
    //     let dp = DynamicProgramBuilder::new()
    //         .simple()
    //         .time_limit(10)
    //         .field_probabilities(fps)
    //         .build();
    //
    //     assert!(matches!(
    //         dp,
    //         Err(DynamicProgramBuilderError::WrongSizeOfFieldProbabilities)
    //     ));
    // }

    #[test]
    fn test_barrier_out_of_range() {
        let dp = DynamicProgramBuilder::new()
            .simple()
            .time_limit(10)
            .add_single_barrier(xy!(25, 5))
            .build();

        assert!(matches!(
            dp,
            Err(DynamicProgramBuilderError::BarrierOutOfRange)
        ));

        let dp = DynamicProgramBuilder::new()
            .simple()
            .time_limit(10)
            .add_single_barrier(xy!(5, 25))
            .build();

        assert!(matches!(
            dp,
            Err(DynamicProgramBuilderError::BarrierOutOfRange)
        ));

        let dp = DynamicProgramBuilder::new()
            .simple()
            .time_limit(10)
            .add_rect_barrier(xy!(15, 5), xy!(25, 5))
            .build();

        assert!(matches!(
            dp,
            Err(DynamicProgramBuilderError::BarrierOutOfRange)
        ));

        let dp = DynamicProgramBuilder::new()
            .simple()
            .time_limit(10)
            .add_rect_barrier(xy!(5, 15), xy!(5, 25))
            .build();

        assert!(matches!(
            dp,
            Err(DynamicProgramBuilderError::BarrierOutOfRange)
        ));
    }

    #[test]
    // fn test_multiple_kernels_for_single() {
    //     let dp = DynamicProgramBuilder::new()
    //         .simple()
    //         .time_limit(10)
    //         .kernels(vec![Kernel::from_generator(SimpleRwGenerator).unwrap(); 10])
    //         .build();
    //
    //     assert!(matches!(
    //         dp,
    //         Err(DynamicProgramBuilderError::MultipleKernelsForSimple)
    //     ));
    //
    //     let dp = DynamicProgramBuilder::new()
    //         .simple()
    //         .time_limit(10)
    //         .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
    //         .kernels(vec![Kernel::from_generator(SimpleRwGenerator).unwrap(); 10])
    //         .build();
    //
    //     assert!(matches!(
    //         dp,
    //         Err(DynamicProgramBuilderError::MultipleKernelsForSimple)
    //     ));
    //
    //     let dp = DynamicProgramBuilder::new()
    //         .simple()
    //         .time_limit(10)
    //         .kernels(vec![Kernel::from_generator(SimpleRwGenerator).unwrap(); 10])
    //         .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
    //         .build();
    //
    //     assert!(matches!(
    //         dp,
    //         Err(DynamicProgramBuilderError::MultipleKernelsForSimple)
    //     ));
    // }
    //
    // #[test]
    // fn test_single_kernel_for_multi() {
    //     let dp = DynamicProgramBuilder::new()
    //         .multi()
    //         .time_limit(10)
    //         .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
    //         .build();
    //
    //     assert!(matches!(
    //         dp,
    //         Err(DynamicProgramBuilderError::SingleKernelForMulti)
    //     ));
    //
    //     let dp = DynamicProgramBuilder::new()
    //         .multi()
    //         .time_limit(10)
    //         .kernels(vec![Kernel::from_generator(SimpleRwGenerator).unwrap(); 10])
    //         .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
    //         .build();
    //
    //     assert!(matches!(
    //         dp,
    //         Err(DynamicProgramBuilderError::SingleKernelForMulti)
    //     ));
    //
    //     let dp = DynamicProgramBuilder::new()
    //         .multi()
    //         .time_limit(10)
    //         .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
    //         .kernels(vec![Kernel::from_generator(SimpleRwGenerator).unwrap(); 10])
    //         .build();
    //
    //     assert!(matches!(
    //         dp,
    //         Err(DynamicProgramBuilderError::SingleKernelForMulti)
    //     ));
    // }
    #[test]
    fn test_no_kernels_set() {
        let dp = DynamicProgramBuilder::new().simple().time_limit(10).build();

        assert!(matches!(dp, Err(DynamicProgramBuilderError::NoKernelsSet)));

        // let dp = DynamicProgramBuilder::new().multi().time_limit(10).build();
        //
        // assert!(matches!(dp, Err(DynamicProgramBuilderError::NoKernelsSet)));
    }

    // #[test]
    // fn test_correct() {
    //     let dp = DynamicProgramBuilder::new()
    //         .with_type(DynamicProgramType::Simple)
    //         .time_limit(10)
    //         .kernel(Kernel::from_generator(SimpleRwGenerator).unwrap())
    //         .field_probabilities(vec![vec![1.0; 21]; 21])
    //         .add_rect_barrier(xy!(5, -5), xy!(5, 5))
    //         .build();
    //
    //     assert!(matches!(dp, Ok(_)));
    //
    //     let dp = DynamicProgramBuilder::new()
    //         .with_type(DynamicProgramType::Multi)
    //         .time_limit(10)
    //         .kernels(
    //             Kernel::multiple_from_generator(CorrelatedRwGenerator { persistence: 0.5 })
    //                 .unwrap(),
    //         )
    //         .field_probabilities(vec![vec![1.0; 21]; 21])
    //         .add_rect_barrier(xy!(5, -5), xy!(5, 5))
    //         .build();
    //
    //     assert!(matches!(dp, Ok(_)));
    // }
}
