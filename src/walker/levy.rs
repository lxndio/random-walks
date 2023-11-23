use crate::dp::DynamicProgramPool;
use crate::walker::{Walk, Walker, WalkerError};
use num::Zero;
use rand::distributions::{WeightedError, WeightedIndex};
use rand::prelude::*;

pub struct LevyWalker {
    pub jump_probability: f64,
    pub jump_distance: usize,
}

impl Walker for LevyWalker {
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

            // Check if jump happens here
            let distance = if thread_rng().gen_range(0f64..1f64) <= self.jump_probability {
                self.jump_distance
            } else {
                1
            };

            let mut prev_probs = vec![
                dp.at(x - distance as isize, y, t - 1), // West
                dp.at(x, y - distance as isize, t - 1), // North
                dp.at(x + distance as isize, y, t - 1), // East
                dp.at(x, y + distance as isize, t - 1), // South
            ];

            // Only allow staying if no jump occurs
            if distance == 1 {
                prev_probs.push(dp.at(x, y, t - 1)); // Stay
            }

            let direction = match WeightedIndex::new(prev_probs) {
                Ok(dist) => dist.sample(&mut rng),
                Err(WeightedError::AllWeightsZero) => return Err(WalkerError::InconsistentPath),
                _ => return Err(WalkerError::RandomDistributionError),
            };

            match direction {
                0 => x -= distance as isize, // West
                1 => y -= distance as isize, // North
                2 => x += distance as isize, // East
                3 => y += distance as isize, // South
                4 => (),                     // Stay
                _ => unreachable!("Other directions should not be chosen from the distribution"),
            }
        }

        path.reverse();
        path.insert(0, (x as i64, y as i64).into());

        Ok(path.into())
    }

    fn name(&self, short: bool) -> String {
        if short {
            String::from("lw")
        } else {
            String::from("Lévy Walker")
        }
    }
}
