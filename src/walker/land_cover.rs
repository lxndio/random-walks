use crate::dp::DynamicProgramPool;
use crate::kernel::Kernel;
use crate::walker::{Walk, Walker, WalkerError};
use num::Zero;
use rand::distributions::{WeightedError, WeightedIndex};
use rand::prelude::*;
use std::collections::HashMap;

#[derive(Clone)]
pub struct LandCoverWalker {
    max_step_sizes: HashMap<usize, usize>,
    field_types: Vec<Vec<usize>>,
    kernels: Vec<Kernel>,
}

impl LandCoverWalker {
    pub fn new(
        max_step_sizes: HashMap<usize, usize>,
        mut field_types: Vec<Vec<usize>>,
        kernels: Vec<(usize, Kernel)>,
    ) -> Self {
        // Map field types to contiguous value range

        let mut kernels_mapped = Vec::new();
        let mut field_type_map = HashMap::new();
        let mut i = 0usize;

        for (field_type, kernel) in kernels.iter() {
            kernels_mapped.push(kernel.clone());
            field_type_map.insert(field_type, i);
            i += 1;
        }

        for x in 0..2 * field_types[0].len() + 1 {
            for y in 0..2 * field_types[0].len() + 1 {
                field_types[x][y] = field_type_map[&field_types[x][y]];
            }
        }

        Self {
            max_step_sizes,
            field_types,
            kernels: kernels_mapped,
        }
    }
}

impl Walker for LandCoverWalker {
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
        let time_limit = (self.field_types.len() / 2) as isize;
        let (mut x, mut y) = (to_x, to_y);
        let mut rng = rand::thread_rng();

        // Check if any path exists leading to the given end point
        if dp.at(to_x, to_y, time_steps).is_zero() {
            return Err(WalkerError::NoPathExists);
        }

        for t in (1..time_steps).rev() {
            path.push((x as i64, y as i64).into());

            let current_land_cover =
                self.field_types[(time_limit + x) as usize][(time_limit + y) as usize];
            let max_step_size = self.max_step_sizes[&current_land_cover] as isize;

            let mut prev_probs = Vec::new();
            let mut movements = Vec::new();

            for i in x - max_step_size..=x + max_step_size {
                for j in y - max_step_size..=y + max_step_size {
                    let p_b = dp.at_or(i, j, t - 1, 0.0);
                    let p_a = dp.at_or(x, y, t, 0.0);
                    let p_a_b = self.kernels[current_land_cover].at(x - i, y - j);

                    prev_probs.push((p_a_b * p_b) / p_a);
                    movements.push((i - x, j - y));
                }
            }

            let direction = match WeightedIndex::new(prev_probs) {
                Ok(dist) => dist.sample(&mut rng),
                Err(WeightedError::AllWeightsZero) => return Err(WalkerError::InconsistentPath),
                _ => return Err(WalkerError::RandomDistributionError),
            };
            let (dx, dy) = movements[direction];

            x += dx;
            y += dy;
        }

        path.reverse();
        path.insert(0, (x as i64, y as i64).into());

        Ok(path.into())
    }

    fn name(&self, short: bool) -> String {
        if short {
            String::from("lcw")
        } else {
            String::from("Land Cover Walker")
        }
    }
}
