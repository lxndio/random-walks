use crate::kernel::generator::{KernelGenerator, KernelGeneratorError};
use crate::kernel::Kernel;
use statrs::distribution::{Continuous, MultivariateNormal};

pub struct NormalDistGenerator {
    pub diffusion: f64,
    pub size: usize,
}

impl NormalDistGenerator {
    pub fn new(diffusion: f64, size: usize) -> Self {
        Self { diffusion, size }
    }
}

impl KernelGenerator for NormalDistGenerator {
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

        let mean = vec![(self.size / 2) as f64, (self.size / 2) as f64];
        let cov = vec![self.diffusion, 0.0, 0.0, self.diffusion];
        let distribution = MultivariateNormal::new(mean, cov).unwrap();

        for x in 0..21 {
            for y in 0..21 {
                kernel.probabilities[x][y] = distribution.pdf(&vec![x as f64, y as f64].into());
            }
        }

        // Normalize values so that they sum up to 1.0
        let sum: f64 = kernel.probabilities.iter().flatten().sum();

        for x in 0..21 {
            for y in 0..21 {
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
