use statrs::distribution::{Continuous, MultivariateNormal};

use crate::dataset::point::XYPoint;
use crate::kernel::generator::{KernelGenerator, KernelGeneratorError};
use crate::kernel::Kernel;

pub enum HalfNormalDistSide {
    Left,
    Right,
    Top,
    Bottom,
}

pub struct HalfNormalDistGenerator {
    pub diffusion: f64,
    pub size: usize,
    pub mean: XYPoint,
    pub side: HalfNormalDistSide,
}

impl HalfNormalDistGenerator {
    pub fn new(diffusion: f64, size: usize, mean: XYPoint, side: HalfNormalDistSide) -> Self {
        Self {
            diffusion,
            size,
            mean,
            side,
        }
    }
}

impl KernelGenerator for HalfNormalDistGenerator {
    fn prepare(&self, kernels: &mut Vec<Kernel>) -> Result<(), KernelGeneratorError> {
        kernels
            .get_mut(0)
            .ok_or(KernelGeneratorError::OneKernelRequired)?
            .initialize(self.size)?;

        Ok(())
    }

    fn generate(&self, kernels: &mut Vec<Kernel>) -> Result<(), KernelGeneratorError> {
        let kernel = kernels
            .get_mut(0)
            .ok_or(KernelGeneratorError::OneKernelRequired)?;

        let mean = vec![
            (self.size / 2 + self.mean.x as usize) as f64,
            (self.size / 2 + self.mean.y as usize) as f64,
        ];
        let cov = vec![self.diffusion, 0.0, 0.0, self.diffusion];
        let distribution = MultivariateNormal::new(mean, cov).unwrap();

        for x in 0..self.size {
            for y in 0..self.size {
                kernel.probabilities[x][y] = distribution.pdf(&vec![x as f64, y as f64].into());
            }
        }

        // Remove values on side that should not be kept
        match self.side {
            HalfNormalDistSide::Left => {
                for x in 0..self.size {
                    for y in 0..self.size {
                        if x > self.size / 2 + 1 {
                            kernel.probabilities[x][y] = 0.0;
                        }
                    }
                }
            }
            HalfNormalDistSide::Right => {
                for x in 0..self.size {
                    for y in 0..self.size {
                        if x < self.size / 2 - 1 {
                            kernel.probabilities[x][y] = 0.0;
                        }
                    }
                }
            }
            HalfNormalDistSide::Top => {
                for x in 0..self.size {
                    for y in 0..self.size {
                        if y > self.size / 2 + 1 {
                            kernel.probabilities[x][y] = 0.0;
                        }
                    }
                }
            }
            HalfNormalDistSide::Bottom => {
                for x in 0..self.size {
                    for y in 0..self.size {
                        if y < self.size / 2 - 1 {
                            kernel.probabilities[x][y] = 0.0;
                        }
                    }
                }
            }
        }

        // Normalize values so that they sum up to 1.0
        let sum: f64 = kernel.probabilities.iter().flatten().sum();

        for x in 0..self.size {
            for y in 0..self.size {
                kernel.probabilities[x][y] /= sum;
            }
        }

        Ok(())
    }

    fn generates_qty(&self) -> usize {
        1
    }

    fn name(&self) -> (String, String) {
        ("nd".into(), "Normal Distribution".into())
    }
}
