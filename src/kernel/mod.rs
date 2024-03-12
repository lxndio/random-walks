//! Provides functionality for creating kernels, as well as pre-defined kernel generators.

use std::fmt::{Debug, Formatter};
use std::ops::{Div, DivAssign, Index, IndexMut, Mul, MulAssign};

use anyhow::bail;
use serde::{Deserialize, Serialize};
use strum::EnumIter;

use crate::kernel::generator::{KernelGenerator, KernelGeneratorError};

pub mod biased_correlated_rw;
pub mod biased_rw;
pub mod correlated_rw;
pub mod generator;
pub mod normal_dist;
pub mod simple_rw;

#[derive(Clone)]
pub struct Kernel {
    pub probabilities: Vec<Vec<f64>>,
    name: (String, String),
}

impl Kernel {
    pub fn try_new(size: usize, name: (String, String)) -> anyhow::Result<Self> {
        if size % 2 == 0 {
            bail!("size must be odd");
        }

        Ok(Self {
            probabilities: vec![vec![0.0; size]; size],
            name,
        })
    }

    pub fn from_generator(generator: impl KernelGenerator) -> Result<Kernel, KernelGeneratorError> {
        let kernel = Kernel {
            probabilities: Vec::new(),
            name: generator.name(),
        };
        let mut kernels = vec![kernel];

        generator.prepare(&mut kernels)?;
        generator.generate(&mut kernels)?;

        Ok(kernels[0].clone())
    }

    pub fn multiple_from_generator(
        generator: impl KernelGenerator,
    ) -> Result<Vec<Kernel>, KernelGeneratorError> {
        let kernel = Kernel {
            probabilities: Vec::new(),
            name: generator.name(),
        };
        let mut kernels = vec![kernel.clone(); generator.generates_qty()];

        generator.prepare(&mut kernels)?;
        generator.generate(&mut kernels)?;

        Ok(kernels)
    }

    pub fn try_from_value(size: usize, value: f64) -> anyhow::Result<Self> {
        if size % 2 == 0 {
            bail!("size must be odd")
        }

        Ok(Self {
            probabilities: vec![vec![value; size]; size],
            name: (String::new(), String::new()),
        })
    }

    pub fn initialize(&mut self, size: usize) -> Result<(), KernelGeneratorError> {
        if size % 2 == 1 {
            self.probabilities = vec![vec![0.0; size]; size];

            Ok(())
        } else {
            Err(KernelGeneratorError::SizeEven)
        }
    }

    pub fn size(&self) -> usize {
        self.probabilities.len()
    }

    pub fn sum(&self) -> f64 {
        let mut sum = 0.0;

        for x in 0..self.size() {
            for y in 0..self.size() {
                sum += self.probabilities[x][y];
            }
        }

        sum
    }

    pub fn set(&mut self, x: isize, y: isize, val: f64) {
        let x = ((self.probabilities.len() / 2) as isize + x) as usize;
        let y = ((self.probabilities.len() / 2) as isize + y) as usize;

        self.probabilities[x][y] = val;
    }

    pub fn at(&self, x: isize, y: isize) -> f64 {
        let x = ((self.probabilities.len() / 2) as isize + x) as usize;
        let y = ((self.probabilities.len() / 2) as isize + y) as usize;

        self.probabilities[x][y]
    }

    /// Rotate kernel matrix clockwise by `degrees`. Only multiples of 90° are supported.
    pub fn rotate(&mut self, degrees: usize) -> Result<(), String> {
        if degrees % 90 != 0 {
            Err("degrees must be a multiple of 90.".into())
        } else {
            let n = self.probabilities.len();

            for _ in 0..degrees / 90 {
                // Source: https://www.enjoyalgorithms.com/blog/rotate-a-matrix-by-90-degrees-in-an-anticlockwise-direction
                for i in 0..n / 2 {
                    for j in i..n - i - 1 {
                        let temp = self.probabilities[i][j];

                        self.probabilities[i][j] = self.probabilities[j][n - 1 - i];
                        self.probabilities[j][n - 1 - i] = self.probabilities[n - 1 - i][n - 1 - j];
                        self.probabilities[n - 1 - i][n - 1 - j] = self.probabilities[n - 1 - j][i];
                        self.probabilities[n - 1 - j][i] = temp;
                    }
                }
            }

            Ok(())
        }
    }

    pub fn name(&self, short: bool) -> String {
        if short {
            self.name.0.clone()
        } else {
            self.name.1.clone()
        }
    }
}

impl Debug for Kernel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut res = String::new();

        for y in 0..self.size() {
            for x in 0..self.size() {
                if x == self.size() / 2 && y == self.size() / 2 {
                    res.remove(res.len() - 1);
                    res += "[";
                }
                if self.probabilities[x][y] == 0.0 {
                    res += "_._____";
                } else {
                    res += &format!("{:.5}", self.probabilities[x][y]);
                }
                if x == self.size() / 2 && y == self.size() / 2 {
                    res += "]";
                } else {
                    res += " ";
                }
            }
            res += "\n";
        }

        f.write_str(res.as_str())
    }
}

impl PartialEq for Kernel {
    fn eq(&self, other: &Self) -> bool {
        self.probabilities == other.probabilities
    }
}

impl Mul for Kernel {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.size() == rhs.size() {
            let mut new_kernel = self.clone();

            for x in 0..self.size() {
                for y in 0..self.size() {
                    new_kernel.probabilities[x][y] *= rhs.probabilities[x][y];
                }
            }

            new_kernel
        } else {
            panic!("both kernels must have the same size for multiplication");
        }
    }
}

impl MulAssign for Kernel {
    fn mul_assign(&mut self, rhs: Self) {
        if self.size() == rhs.size() {
            for x in 0..self.size() {
                for y in 0..self.size() {
                    self.probabilities[x][y] *= rhs.probabilities[x][y];
                }
            }
        } else {
            panic!("both kernels must have the same size for multiplication");
        }
    }
}

impl Div for Kernel {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        if self.size() == rhs.size() {
            let mut new_kernel = self.clone();

            for x in 0..self.size() {
                for y in 0..self.size() {
                    new_kernel.probabilities[x][y] /= rhs.probabilities[x][y];
                }
            }

            new_kernel
        } else {
            panic!("both kernels must have the same size for division");
        }
    }
}

impl DivAssign for Kernel {
    fn div_assign(&mut self, rhs: Self) {
        if self.size() == rhs.size() {
            for x in 0..self.size() {
                for y in 0..self.size() {
                    self.probabilities[x][y] /= rhs.probabilities[x][y];
                }
            }
        } else {
            panic!("both kernels must have the same size for division");
        }
    }
}

/// A macro that allows quick creation of a custom kernel.
#[macro_export]
macro_rules! kernel {
    ($($x:expr),+ $(,)?) => {{
        let probs = vec![$($x),*];
        let size = (probs.len() as f64).sqrt() as usize;

        if size.pow(2) != probs.len() {
            panic!("kernel! needs n^2 elements to create a kernel of size n");
        }

        let mut kernel = Kernel::try_new(size, ("ck".into(), "Custom Kernel".into())).unwrap();

        kernel.probabilities = (0..size)
            .map(|y| probs.iter().skip(y).step_by(size).copied().collect())
            .collect();

        kernel
    }}
}

#[derive(Default, Debug, PartialEq, Copy, Clone, EnumIter, Serialize, Deserialize)]
pub enum Direction {
    North,
    East,
    South,
    West,
    #[default]
    Stay,
}

#[derive(Default, Debug)]
pub struct Directions<T> {
    pub north: T,
    pub east: T,
    pub south: T,
    pub west: T,
    pub stay: T,
}

impl TryFrom<(isize, isize)> for Direction {
    type Error = &'static str;

    fn try_from(value: (isize, isize)) -> Result<Self, Self::Error> {
        match value {
            (0, -1) => Ok(Self::North),
            (1, 0) => Ok(Self::East),
            (0, 1) => Ok(Self::South),
            (-1, 0) => Ok(Self::West),
            (0, 0) => Ok(Self::Stay),
            _ => Err("Invalid direction"),
        }
    }
}

impl From<Direction> for (isize, isize) {
    fn from(value: Direction) -> Self {
        match value {
            Direction::North => (0, -1),
            Direction::East => (1, 0),
            Direction::South => (0, 1),
            Direction::West => (-1, 0),
            Direction::Stay => (0, 0),
        }
    }
}

impl<T: Default> Directions<T> {
    pub fn new() -> Self {
        Self {
            north: Default::default(),
            east: Default::default(),
            south: Default::default(),
            west: Default::default(),
            stay: Default::default(),
        }
    }
}

impl<T> Index<Direction> for Directions<T> {
    type Output = T;

    fn index(&self, direction: Direction) -> &Self::Output {
        match direction {
            Direction::North => &self.north,
            Direction::East => &self.east,
            Direction::South => &self.south,
            Direction::West => &self.west,
            Direction::Stay => &self.stay,
        }
    }
}

impl<T> IndexMut<Direction> for Directions<T> {
    fn index_mut(&mut self, direction: Direction) -> &mut Self::Output {
        match direction {
            Direction::North => &mut self.north,
            Direction::East => &mut self.east,
            Direction::South => &mut self.south,
            Direction::West => &mut self.west,
            Direction::Stay => &mut self.stay,
        }
    }
}

#[cfg(test)]
#[rustfmt::skip]
mod tests {
    use crate::kernel::Kernel;

    #[test]
    fn test_rotate_invalid() {
        let mut kernel = kernel![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];

        assert!(kernel.rotate(87).is_err());
    }

    #[test]
    fn test_rotate_90() {
        let mut kernel = kernel![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];

        let mut correct_rotation = kernel![
            7.0, 4.0, 1.0,
            8.0, 5.0, 2.0,
            9.0, 6.0, 3.0,
        ];

        assert!(kernel.rotate(90).is_ok());
        assert_eq!(kernel, correct_rotation);
    }

    #[test]
    fn test_rotate_180() {
        let mut kernel = kernel![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];

        let mut correct_rotation = kernel![
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        ];

        assert!(kernel.rotate(180).is_ok());
        assert_eq!(kernel, correct_rotation);
    }

    #[test]
    fn test_rotate_270() {
        let mut kernel = kernel![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];

        let mut correct_rotation = kernel![
            3.0, 6.0, 9.0,
            2.0, 5.0, 8.0,
            1.0, 4.0, 7.0,
        ];

        assert!(kernel.rotate(270).is_ok());
        assert_eq!(kernel, correct_rotation);
    }

    #[test]
    fn test_kernel_macro() {
        let kernel = kernel![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];

        let kernel_correct = Kernel {
            probabilities: vec![
                vec![1.0, 4.0, 7.0],
                vec![2.0, 5.0, 8.0],
                vec![3.0, 6.0, 9.0],
            ],
            name: ("".into(), "".into()),
        };

        assert_eq!(kernel, kernel_correct);
    }
}
