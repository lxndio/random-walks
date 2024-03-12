use strum::IntoEnumIterator;

use crate::kernel::generator::{KernelGenerator, KernelGeneratorError};
use crate::kernel::{Direction, Kernel};

pub struct BiasedRwGenerator {
    pub probability: f64,
    pub direction: Direction,
}

impl KernelGenerator for BiasedRwGenerator {
    fn prepare(&self, kernels: &mut Vec<Kernel>) -> Result<(), KernelGeneratorError> {
        kernels
            .get_mut(0)
            .ok_or(KernelGeneratorError::OneKernelRequired)?
            .initialize(3)
            .unwrap();

        Ok(())
    }

    fn generate(&self, kernels: &mut Vec<Kernel>) -> Result<(), KernelGeneratorError> {
        let kernel = kernels
            .get_mut(0)
            .ok_or(KernelGeneratorError::OneKernelRequired)?;
        let (direction_x, direction_y) = self.direction.into();
        let other_prob = (1.0 - self.probability) / 4.0;

        kernel.set(direction_x, direction_y, self.probability);

        for direction in Direction::iter() {
            if direction != self.direction {
                let (direction_x, direction_y) = direction.into();

                kernel.set(direction_x, direction_y, other_prob);
            }
        }

        Ok(())
    }

    fn generates_qty(&self) -> usize {
        1
    }

    fn name(&self) -> (String, String) {
        ("brw".into(), "Biased RW".into())
    }
}

#[cfg(test)]
mod tests {
    use crate::kernel;
    use crate::kernel::biased_rw::BiasedRwGenerator;
    use crate::kernel::{Direction, Kernel};

    #[test]
    #[rustfmt::skip]
    fn test_biased_rw() {
        let kernel = Kernel::from_generator(BiasedRwGenerator {
            probability: 0.5,
            direction: Direction::North,
        });

        let kernel_correct = kernel![
            0.0,   0.5,   0.0,
            0.125, 0.125, 0.125,
            0.0,   0.125, 0.0
        ];

        assert!(kernel.is_ok());
        assert_eq!(kernel.unwrap(), kernel_correct);
    }
}
