use strum::IntoEnumIterator;

use crate::kernel::biased_rw::BiasedRwGenerator;
use crate::kernel::generator::{KernelGenerator, KernelGeneratorError};
use crate::kernel::{Direction, Kernel};

pub struct CorrelatedRwGenerator {
    pub persistence: f64,
}

impl KernelGenerator for CorrelatedRwGenerator {
    fn prepare(&self, kernels: &mut Vec<Kernel>) -> Result<(), KernelGeneratorError> {
        if kernels.len() != self.generates_qty() {
            Err(KernelGeneratorError::NotEnoughKernels)
        } else {
            for kernel in kernels.iter_mut() {
                kernel.initialize(3).unwrap();
            }

            Ok(())
        }
    }

    fn generate(&self, kernels: &mut Vec<Kernel>) -> Result<(), KernelGeneratorError> {
        if kernels.len() != self.generates_qty() {
            Err(KernelGeneratorError::NotEnoughKernels)
        } else {
            for (i, direction) in Direction::iter().enumerate() {
                kernels[i] = Kernel::from_generator(BiasedRwGenerator {
                    probability: self.persistence,
                    direction,
                })
                .unwrap();
            }

            Ok(())
        }
    }

    fn generates_qty(&self) -> usize {
        5
    }

    fn name(&self) -> (String, String) {
        ("brw".into(), "Biased RW".into())
    }
}

#[cfg(test)]
mod tests {
    use crate::kernel;
    use crate::kernel::correlated_rw::CorrelatedRwGenerator;
    use crate::kernel::Kernel;

    #[test]
    #[rustfmt::skip]
    fn test_correlated_rw() {
        let kernels = Kernel::multiple_from_generator(CorrelatedRwGenerator { persistence: 0.5 });

        let kernel_correct_0 = kernel![
            0.0,   0.5,   0.0,
            0.125, 0.125, 0.125,
            0.0,   0.125, 0.0
        ];
        let kernel_correct_1 = kernel![
            0.0,   0.125, 0.0,
            0.125, 0.125, 0.5,
            0.0,   0.125, 0.0
        ];
        let kernel_correct_2 = kernel![
            0.0,   0.125, 0.0,
            0.125, 0.125, 0.125,
            0.0,   0.5,   0.0
        ];
        let kernel_correct_3 = kernel![
            0.0, 0.125, 0.0,
            0.5, 0.125, 0.125,
            0.0, 0.125, 0.0
        ];
        let kernel_correct_4 = kernel![
            0.0,   0.125, 0.0,
            0.125, 0.5,   0.125,
            0.0,   0.125, 0.0
        ];

        assert!(kernels.is_ok());

        let kernels = kernels.unwrap();

        assert_eq!(kernels[0], kernel_correct_0);
        assert_eq!(kernels[1], kernel_correct_1);
        assert_eq!(kernels[2], kernel_correct_2);
        assert_eq!(kernels[3], kernel_correct_3);
        assert_eq!(kernels[4], kernel_correct_4);
    }
}
