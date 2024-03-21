use log::{debug, error, trace};
use num::Zero;
use rand::distributions::{WeightedError, WeightedIndex};
use rand::prelude::Distribution;
use rand::Rng;

use crate::dp::simple::DynamicProgram;
use crate::dp::DynamicProgramPool;
use crate::kernel::Kernel;
use crate::walker::{Walk, Walker, WalkerError};

pub struct CorrelatedMultiStepWalker {
    max_step_size: usize,
    kernels: Vec<Kernel>,
    directions_per_axis: usize,
}

impl CorrelatedMultiStepWalker {
    pub fn new(max_step_size: usize, kernels: Vec<Kernel>, directions_per_axis: usize) -> Self {
        Self {
            max_step_size,
            kernels,
            directions_per_axis,
        }
    }
}

impl Walker for CorrelatedMultiStepWalker {
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

        let max_step_size = self.max_step_size as isize;
        let mut path = Vec::new();
        let (mut x, mut y) = (to_x, to_y);
        let mut rng = rand::thread_rng();

        // Divide into grid sections
        let mut sections = Vec::new();
        let mut start = -max_step_size;
        let section_size = (2 * max_step_size + 1) / self.directions_per_axis as isize;
        let remainder = (2 * max_step_size + 1) % self.directions_per_axis as isize;

        for _ in 0..self.directions_per_axis {
            let end = start + section_size;
            sections.push(start..end);
            start = end;
        }

        if remainder != 0 {
            let middle = sections.len() / 2;
            let middle_section = &mut sections[middle];
            middle_section.end += remainder;
        }

        debug!("Sections: {:?}", sections);

        // Check if any path exists leading to the given end point for any variant
        if !(0..dp_qty).any(|i| !dp.at(to_x, to_y, time_steps, i).unwrap().is_zero()) {
            return Err(WalkerError::NoPathExists);
        }

        path.push((x as i64, y as i64).into());

        // Compute first (= last, because reconstructing backwards) step manually
        let direction = rng.gen_range(0..9);

        match direction {
            0 => {
                x -= 1;
                y -= 1;
            }
            1 => y -= 1,
            2 => {
                x += 1;
                y -= 1;
            }
            3 => x -= 1,
            4 => (),
            5 => x += 1,
            6 => {
                x -= 1;
                y += 1;
            }
            7 => y += 1,
            8 => {
                x += 1;
                y += 1;
            }
            _ => unimplemented!(),
        }

        let mut last_direction = direction;
        debug!("First direction: {}", last_direction);

        for t in (1..time_steps).rev() {
            debug!("Time step: {}", t);

            path.push((x as i64, y as i64).into());

            let mut prev_probs = Vec::new();
            let mut movements = Vec::new();

            for i in x - max_step_size..=x + max_step_size {
                for j in y - max_step_size..=y + max_step_size {
                    let p_b = dp.at_or(i, j, t - 1, last_direction, 0.0).unwrap();
                    let p_a = dp.at_or(x, y, t, last_direction, 0.0).unwrap();
                    let p_a_b = self.kernels[last_direction].at(i - x, j - y);

                    trace!(
                        "p_b: {}, p_a: {}, p_a_b: {}, prob: {}",
                        p_b,
                        p_a,
                        p_a_b,
                        (p_a_b * p_b) / p_a
                    );

                    prev_probs.push((p_a_b * p_b) / p_a);
                    movements.push((i - x, j - y));
                }
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
            let (dx, dy) = movements[direction];

            x += dx;
            y += dy;

            debug!("Movement: ({}, {})", dx, dy);

            let row = sections
                .iter()
                .position(|section| section.contains(&dx))
                .unwrap();
            let column = sections
                .iter()
                .position(|section| section.contains(&dy))
                .unwrap()
                * self.directions_per_axis;
            last_direction = row + column;
            debug!("Last direction: {}", last_direction);
        }

        path.reverse();
        path.insert(0, (x as i64, y as i64).into());

        Ok(path.into())
    }

    fn name(&self, short: bool) -> String {
        if short {
            String::from("cmsw")
        } else {
            String::from("Correlated Multi Step Walker")
        }
    }
}
