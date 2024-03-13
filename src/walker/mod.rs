//! Provides walkers used to generate random walks by using a dynamic program.

pub mod correlated;
pub mod correlated_multi_step;
pub mod land_cover;
pub mod levy;
pub mod multi_step;
pub mod standard;

use thiserror::Error;

use crate::dp::DynamicProgramPool;
use crate::walk::Walk;

pub trait Walker {
    fn generate_path(
        &self,
        dp: &DynamicProgramPool,
        to_x: isize,
        to_y: isize,
        time_steps: usize,
    ) -> Result<Walk, WalkerError>;

    fn generate_paths(
        &self,
        dp: &DynamicProgramPool,
        qty: usize,
        to_x: isize,
        to_y: isize,
        time_steps: usize,
    ) -> Result<Vec<Walk>, WalkerError> {
        let mut paths = Vec::new();

        for _ in 0..qty {
            paths.push(self.generate_path(dp, to_x, to_y, time_steps)?);
        }

        Ok(paths)
    }

    fn name(&self, short: bool) -> String;
}

#[derive(Error, Debug)]
pub enum WalkerError {
    #[error("the walker requires a single dynamic program but multiple were given")]
    RequiresSingleDynamicProgram,

    #[error("the walker requires multiple dynamic programs but only a single one was given")]
    RequiresMultipleDynamicPrograms,

    #[error("no path exists")]
    NoPathExists,

    #[error("found an inconsistent path, probably due to wrong settings in the dynamic program or walker")]
    InconsistentPath,

    #[error("error while computing random distribution")]
    RandomDistributionError,
}
