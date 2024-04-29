use log::{debug, error, trace};
use num::Zero;
use rand::distributions::{WeightedError, WeightedIndex};
use rand::prelude::Distribution;
use rand::Rng;

use crate::dp::DynamicProgramPool;
use crate::kernel::Kernel;
use crate::walker::{Walk, Walker, WalkerError};

pub struct CorrelatedFixedStepWalker {
    step_size: usize,
    kernels: Vec<Kernel>,
}

impl CorrelatedFixedStepWalker {
    pub fn new(step_size: usize, kernels: Vec<Kernel>) -> Self {
        Self { step_size, kernels }
    }
}

impl Walker for CorrelatedFixedStepWalker {
    fn generate_path(
        &self,
        dp: &DynamicProgramPool,
        to_x: isize,
        to_y: isize,
        time_steps: usize,
    ) -> Result<Walk, WalkerError> {
        if matches!(dp, DynamicProgramPool::Single(_)) {
            return Err(WalkerError::RequiresMultipleDynamicPrograms);
        }

        debug!(
            "Generating path for ({}, {}) with {} time steps",
            to_x, to_y, time_steps
        );

        let dp_qty = match dp {
            DynamicProgramPool::Multiple(dp) => dp.len(),
            DynamicProgramPool::MultipleFromDisk(dp) => dp.len(),
            _ => return Err(WalkerError::RequiresMultipleDynamicPrograms),
        };

        let step_size = self.step_size as isize;
        let mut path = Vec::new();
        let (mut x, mut y) = (to_x, to_y);
        let mut rng = rand::thread_rng();

        // Compute possible fields with given distance from center
        let mut possible_fields = Vec::new();

        for x in -step_size..=step_size {
            for y in -step_size..=step_size {
                if x.abs() + y.abs() == step_size {
                    possible_fields.push((x, y));
                }
            }
        }

        // Check if any path exists leading to the given end point for any variant
        if !(0..dp_qty).any(|i| !dp.at(to_x, to_y, time_steps, i).unwrap().is_zero()) {
            return Err(WalkerError::NoPathExists);
        }

        path.push((x as i64, y as i64).into());

        // Compute first (= last, because reconstructing backwards) step manually
        let direction = rng.gen_range(0..possible_fields.len());

        let (dx, dy) = possible_fields[direction];
        x += dx;
        y += dy;

        let mut last_direction = direction;
        debug!("First direction: {}", last_direction);

        for t in (2..time_steps).rev() {
            debug!("Time step: {}", t);

            path.push((x as i64, y as i64).into());

            let mut prev_probs = Vec::new();

            for (i, j) in possible_fields.iter() {
                // Check if the field is within bounds
                // if x + i < 0 || x + i > (2 * time_steps + 1) as isize || y + j < 0 || y + j > (2 * time_steps + 1) as isize {
                //     prev_probs.push(0.0);
                //     continue;
                // }

                let (i, j) = (x + i, y + j);
                let p_b = dp.at_or(i, j, t - 1, last_direction, 0.0).unwrap();
                let p_a = dp.at_or(x, y, t, last_direction, 0.0).unwrap();
                let p_a_b = self.kernels[last_direction].at(i - x, j - y);

                trace!(
                    "i: {}, j: {}, p_b: {}, p_a: {}, p_a_b: {}, prob: {}",
                    i,
                    j,
                    p_b,
                    p_a,
                    p_a_b,
                    (p_a_b * p_b) / p_a
                );

                prev_probs.push((p_a_b * p_b) / p_a);
            }

            let direction = match WeightedIndex::new(&prev_probs) {
                Ok(dist) => dist.sample(&mut rng),
                Err(WeightedError::AllWeightsZero) => {
                    error!("time step: {t}, x: {x}, y: {y}");
                    return Err(WalkerError::InconsistentPath);
                }
                Err(e) => {
                    error!("Random distribution error: {e}");
                    error!("Weights:\n{:#?}", prev_probs);
                    return Err(WalkerError::RandomDistributionError);
                }
            };
            let (dx, dy) = possible_fields[direction];
            x += dx;
            y += dy;
            last_direction = direction;

            debug!(
                "Direction: {}, Movement: ({}, {}), New Position: ({}, {})",
                direction, dx, dy, x, y
            );
        }

        path.reverse();
        path.insert(0, (x as i64, y as i64).into());

        Ok(path.into())
    }

    fn name(&self, short: bool) -> String {
        if short {
            String::from("cfsw")
        } else {
            String::from("Correlated Fixed Step Walker")
        }
    }
}
