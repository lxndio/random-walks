use num::Zero;
use rand::distributions::{WeightedError, WeightedIndex};
use rand::prelude::*;

use crate::dp::DynamicProgramPool;
use crate::kernel::Kernel;
use crate::walker::{Walk, Walker, WalkerError};

pub struct StandardWalker {
    pub kernel: Kernel,
}

impl Walker for StandardWalker {
    fn generate_path(
        &self,
        dp: &DynamicProgramPool,
        to_x: isize,
        to_y: isize,
        time_steps: usize,
    ) -> Result<Walk, WalkerError> {
        let DynamicProgramPool::Single(dp) = dp else {
            return Err(WalkerError::RequiresSingleDynamicProgram);
        };

        let mut path = Vec::new();
        let (mut x, mut y) = (to_x, to_y);
        let mut rng = rand::thread_rng();

        // Check if any path exists leading to the given end point
        if dp.at(to_x, to_y, time_steps).is_zero() {
            return Err(WalkerError::NoPathExists);
        }

        for t in (1..time_steps).rev() {
            path.push((x as i64, y as i64).into());

            let neighbors = [(0, 0), (-1, 0), (0, -1), (1, 0), (0, 1)];
            let mut prev_probs = Vec::new();

            for (mov_x, mov_y) in neighbors.iter() {
                let (i, j) = (x + mov_x, y + mov_y);

                let p_b = dp.at_or(i, j, t - 1, 0.0);
                let p_a = dp.at_or(x, y, t, 0.0);
                let p_a_b = self.kernel.at(i - x, j - y);

                prev_probs.push((p_a_b * p_b) / p_a);
            }

            let direction = match WeightedIndex::new(prev_probs) {
                Ok(dist) => dist.sample(&mut rng),
                Err(WeightedError::AllWeightsZero) => return Err(WalkerError::InconsistentPath),
                _ => return Err(WalkerError::RandomDistributionError),
            };

            match direction {
                0 => (),     // Stay
                1 => x -= 1, // West
                2 => y -= 1, // North
                3 => x += 1, // East
                4 => y += 1, // South
                _ => unreachable!("Other directions should not be chosen from the distribution"),
            }
        }

        path.reverse();
        path.insert(0, (x as i64, y as i64).into());

        Ok(path.into())
    }

    fn name(&self, short: bool) -> String {
        if short {
            String::from("swg")
        } else {
            String::from("Standard Walker")
        }
    }
}
