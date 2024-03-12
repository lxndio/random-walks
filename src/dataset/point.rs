//! Provides different formats for two-dimensional points.

use std::ops::{Add, Sub};

use num::Signed;

/// Specifies points that have an X- and Y-coordinate.
pub trait Coordinates<T: Signed> {
    fn x(&self) -> T;
    fn y(&self) -> T;
}

/// A 2d-point in geographic coordinate system (GCS).
#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct GCSPoint {
    pub x: f64,
    pub y: f64,
}

impl Coordinates<f64> for GCSPoint {
    fn x(&self) -> f64 {
        self.x
    }

    fn y(&self) -> f64 {
        self.y
    }
}

impl TryFrom<Point> for GCSPoint {
    type Error = ();

    fn try_from(value: Point) -> Result<Self, Self::Error> {
        match value {
            Point::GCS(p) => Ok(p),
            Point::XY(_) => Err(()),
        }
    }
}

impl From<(f64, f64)> for GCSPoint {
    fn from(value: (f64, f64)) -> Self {
        Self {
            x: value.0,
            y: value.1,
        }
    }
}

impl From<GCSPoint> for (f64, f64) {
    fn from(value: GCSPoint) -> Self {
        (value.x, value.y)
    }
}

impl Add for GCSPoint {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for GCSPoint {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl ToString for GCSPoint {
    fn to_string(&self) -> String {
        format!("({}, {})", self.x, self.y)
    }
}

/// A 2d-point in XY coordinate system.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct XYPoint {
    pub x: i64,
    pub y: i64,
}

impl Coordinates<i64> for XYPoint {
    fn x(&self) -> i64 {
        self.x
    }

    fn y(&self) -> i64 {
        self.y
    }
}

impl TryFrom<Point> for XYPoint {
    type Error = ();

    fn try_from(value: Point) -> Result<Self, Self::Error> {
        match value {
            Point::GCS(_) => Err(()),
            Point::XY(p) => Ok(p),
        }
    }
}

impl From<(i64, i64)> for XYPoint {
    fn from(value: (i64, i64)) -> Self {
        Self {
            x: value.0,
            y: value.1,
        }
    }
}

impl From<XYPoint> for (i64, i64) {
    fn from(value: XYPoint) -> Self {
        (value.x, value.y)
    }
}

impl Add for XYPoint {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl Sub for XYPoint {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl ToString for XYPoint {
    fn to_string(&self) -> String {
        format!("({}, {})", self.x, self.y)
    }
}

/// A macro that allows quick creation of an [`XYPoint`](XYPoint).
#[macro_export]
macro_rules! xy {
    ($x:expr, $y:expr) => {
        XYPoint { x: $x, y: $y }
    };
}

/// A 2d-point in either GCS or XY coordinates.
#[derive(Debug, Clone, PartialEq)]
pub enum Point {
    /// A 2d-point in geographic coordinate system (GCS).
    GCS(GCSPoint),

    /// A 2d-point in XY coordinate system.
    XY(XYPoint),
}

impl Default for Point {
    fn default() -> Self {
        Point::GCS(GCSPoint::default())
    }
}

impl Coordinates<f64> for Point {
    fn x(&self) -> f64 {
        match self {
            Self::GCS(point) => point.x,
            Self::XY(_) => panic!("XY points use i64 instead of f64."),
        }
    }

    fn y(&self) -> f64 {
        match self {
            Self::GCS(point) => point.y,
            Self::XY(_) => panic!("XY points use i64 instead of f64."),
        }
    }
}

impl Coordinates<i64> for Point {
    fn x(&self) -> i64 {
        match self {
            Self::GCS(_) => panic!("GCS points use f64 instead of i64."),
            Self::XY(point) => point.x,
        }
    }

    fn y(&self) -> i64 {
        match self {
            Self::GCS(_) => panic!("GCS points use f64 instead of i64."),
            Self::XY(point) => point.y,
        }
    }
}

impl From<GCSPoint> for Point {
    fn from(value: GCSPoint) -> Self {
        Point::GCS(value)
    }
}

impl From<XYPoint> for Point {
    fn from(value: XYPoint) -> Self {
        Point::XY(value)
    }
}

impl ToString for Point {
    fn to_string(&self) -> String {
        match self {
            Point::GCS(p) => format!("GCS{}", p.to_string()),
            Point::XY(p) => format!("XY{}", p.to_string()),
        }
    }
}
