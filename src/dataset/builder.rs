//! Provides a builder for datasets.
//!
//! The [`DatasetBuilder`](DatasetBuilder) allows to create datasets and fill them with either
//! generated or loaded [`Datapoint`s](crate::dataset::Datapoint). Currently it supports the
//! following methods of acquiring data:
//!
//! - Loading from CSV using [`from_csv()`](DatasetBuilder::from_csv)
//! - Loading from a Polars `DataFrame` using [`from_polars()`](DatasetBuilder::from_polars)
//! - Adding points manually using [`add_point()`](DatasetBuilder::add_point) or
//! [`add_points()`](DatasetBuilder::add_points)
//! - Add a line of points using [`line()`](DatasetBuilder::line)
//! - Add points in a certain area using [`fill()`](DatasetBuilder::fill)
//! - Add points to randomly generated locations using [`random()`](DatasetBuilder::random)
//!
//! [`ColumnAction`s](loader::ColumnAction) are used to define which column of the imported data
//! (for CSV and Polars) contains which information, such as the X- and Y coordinates etc.
//!
//! The [`CoordinateType`](loader::CoordinateType) must be specified using
//! [`coordinate_type()`](DatasetBuilder::coordinate_type). It can either be `GCS` for floating
//! point geographical coordinates, or `XY` for integer coordinates in a
//! [Cartesian coordinate system](https://en.wikipedia.org/wiki/Cartesian_coordinate_system).
//!
//! # CSV Files
//!
//! When using [`from_csv()`](DatasetBuilder::from_csv), a column delimiter can be specified using
//! the [`delimiter()`](DatasetBuilder::delimiter) function. Additionally, if the file contains
//! a header row that should be skipped, [`with_header()`](DatasetBuilder::with_header) can be set.
//! Both are optional. The delimiter defaults to a comma, while header turned off by default.
//!
//! # Examples
//!
//! This example loads a dataset from a CSV file. The column actions specify that the first column
//! of the CSV file contains the X-coordinates, the second column contains the Y-coordinates, and
//! the third column contains agent IDs that should be stored as metadata under the key `agent_id`.
//! The coordinate type should be `XY`.
//!
//! ```
//! use randomwalks_lib::dataset::builder::DatasetBuilder;
//! use randomwalks_lib::dataset::loader::{ColumnAction, CoordinateType};
//! use randomwalks_lib::walker::standard::StandardWalker;
//!
//! let dataset = DatasetBuilder::new()
//!     .from_csv("dataset.csv")
//!     .add_column_actions(vec![
//!         ColumnAction::KeepX,
//!         ColumnAction::KeepY,
//!         ColumnAction::KeepMetadata("agent_id")
//!     ])
//!     .coordinate_type(CoordinateType::XY)
//!     .build()
//!     .unwrap();
//!
//! ```
//!
//! The next example creates a dataset without loading points from any source. Instead, 100 points
//! are generated at random locations in between `(0, 0)` and `(500, 500)`.
//!
//! ```
//! use randomwalks_lib::dataset::builder::DatasetBuilder;
//! use randomwalks_lib::dataset::loader::CoordinateType;
//! use randomwalks_lib::xy;
//!
//! let dataset = DatasetBuilder::new()
//!     .coordinate_type(CoordinateType::XY)
//!     .random(100, xy!(0, 0), xy!(500, 500))
//!     .build()
//!     .unwrap();
//! ```
//!

use std::collections::HashMap;

use anyhow::bail;
#[cfg(feature = "polars_loading")]
use polars::prelude::DataFrame;
use rand::Rng;
use thiserror::Error;

use crate::dataset::loader::csv::{CSVLoader, CSVLoaderOptions};
use crate::dataset::loader::polars::{PolarsLoader, PolarsLoaderOptions};
use crate::dataset::loader::{ColumnAction, CoordinateType};
use crate::dataset::point::{Point, XYPoint};
use crate::dataset::{Datapoint, Dataset};
use crate::xy;

/// An error that can occur when using a [`DatasetBuilder`](DatasetBuilder).
#[derive(Error, Debug)]
pub enum DatasetBuilderError {
    /// This error occurs when no source for the dataset has been set. See the documentation of the
    /// [`builder`](crate::dataset::builder) module for a list of possible sources.
    #[error("a data source for the dataset must be set")]
    NoDatasetSourceSet,

    /// This error occurs when no coordinate type has been set. See
    /// [`CoordinateType`](loader::CoordinateType) for possible types. You can set the type using
    /// the [`coordinate_type()`](DatasetBuilder::coordinate_type) function.
    #[error("a coordinate type must be set")]
    NoCoordinateTypeSet,
}

#[derive(Default)]
enum DatasetSource {
    Csv(String),
    #[cfg(feature = "polars_loading")]
    Polars(DataFrame),
    Manual,
    #[default]
    None,
}

/// A builder for datasets that can create datasets from different sources.
///
/// For a detailed description and examples see the documentation of the
/// [`builder`](crate::dataset::builder) module.
pub struct DatasetBuilder {
    source: DatasetSource,
    csv_delimiter: u8,
    csv_header: bool,
    column_actions: Vec<ColumnAction<String>>,
    coordinate_type: Option<CoordinateType>,
    points: Vec<Point>,
}

impl DatasetBuilder {
    /// Creates a new [`DatasetBuilder`].
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    /// Loads data from a CSV file.
    pub fn from_csv<S>(mut self, path: S) -> Self
    where
        S: Into<String>,
    {
        self.source = DatasetSource::Csv(path.into());

        self
    }

    /// Loads data from a Polars `DataFrame`.
    #[cfg(feature = "polars_loading")]
    pub fn from_polars(mut self, df: DataFrame) -> Self {
        self.source = DatasetSource::Polars(df);

        self
    }

    /// Sets the CSV column delimiter.
    pub fn delimiter(mut self, delimiter: u8) -> Self {
        self.csv_delimiter = delimiter;

        self
    }

    /// Turns on a header row for CSV.
    ///
    /// If set, the first row of a CSV file will be skipped.
    pub fn with_header(mut self) -> Self {
        self.csv_header = true;

        self
    }

    /// Adds a [`ColumnAction`](loader::ColumnAction).
    pub fn add_column_action(mut self, column_action: ColumnAction<&str>) -> Self {
        self.column_actions.push(column_action.into());

        self
    }

    /// Adds [`ColumnAction`s](loader::ColumnAction).
    pub fn add_column_actions(mut self, column_actions: Vec<ColumnAction<&str>>) -> Self {
        let mut column_actions: Vec<_> = column_actions.iter().map(|x| x.clone().into()).collect();

        self.column_actions.append(&mut column_actions);

        self
    }

    /// Sets the [`CoordinateType`](crate::dataset::CoordinateType). This must be set.
    pub fn coordinate_type(mut self, coordinate_type: CoordinateType) -> Self {
        self.coordinate_type = Some(coordinate_type);

        self
    }

    /// Adds a point to the dataset.
    pub fn add_point(mut self, point: Point) -> Self {
        self.source = DatasetSource::Manual;
        self.points.push(point);

        self
    }

    /// Adds points to the dataset.
    pub fn add_points(mut self, points: Vec<Point>) -> Self {
        self.source = DatasetSource::Manual;
        self.points.append(&mut points.clone());

        self
    }

    /// Adds a line of points to the dataset.
    ///
    /// This adds `qty` points to the dataset. The first point will be placed at `from`,
    /// with all additional points being spaced by `spacing` from the last point.
    pub fn line(mut self, qty: usize, from: XYPoint, spacing: XYPoint) -> Self {
        self.source = DatasetSource::Manual;

        for i in 0..qty as i64 {
            self.points.push(Point::XY(xy!(
                from.x + spacing.x * i,
                from.y + spacing.y * i
            )));
        }

        self
    }

    /// Adds points filling a space to the dataset.
    ///
    /// This add points to the dataset that are regularly spaced by `spacing` and fill the
    /// area in between `from` and `to`.
    pub fn fill(mut self, from: XYPoint, to: XYPoint, spacing: XYPoint) -> Self {
        self.source = DatasetSource::Manual;

        let (mut x, mut y) = from.into();

        while y < to.y {
            while x < to.x {
                self.points.push(Point::XY(xy!(x, y)));

                x += spacing.x;
            }

            y += spacing.y;
        }

        self
    }

    /// Adds randomly positioned points to the dataset.
    ///
    /// This adds `qty` points with random locations to the dataset. All points are placed in
    /// between `from` and `to`.
    pub fn random(mut self, qty: usize, from: XYPoint, to: XYPoint) -> Self {
        self.source = DatasetSource::Manual;

        let mut rng = rand::thread_rng();

        for _ in 0..qty {
            let x = rng.gen_range(from.x..to.x);
            let y = rng.gen_range(from.y..to.y);

            self.points.push(Point::XY(xy!(x, y)));
        }

        self
    }

    /// Builds a dataset.
    ///
    /// This builds the dataset after all options have been specified. Returns a
    /// [`Dataset`](crate::dataset::Dataset) if successful.
    ///
    /// # Errors
    ///
    /// Returns a [`DatasetBuilderError`] if misconfigured. If loading fails, it returns a
    /// [`DatasetLoaderError`](loader::DatasetLoaderError) or a [`csv::Error`](csv::Error).
    pub fn build(self) -> anyhow::Result<Dataset> {
        let Some(coordinate_type) = self.coordinate_type else {
            bail!(DatasetBuilderError::NoCoordinateTypeSet);
        };

        match self.source {
            DatasetSource::Csv(path) => {
                let loader = CSVLoader::new(CSVLoaderOptions {
                    path,
                    delimiter: self.csv_delimiter,
                    header: self.csv_header,
                    column_actions: self.column_actions,
                    coordinate_type,
                });

                Dataset::from_loader(loader)
            }
            #[cfg(feature = "polars_loading")]
            DatasetSource::Polars(df) => {
                let loader = PolarsLoader::new(PolarsLoaderOptions {
                    df,
                    column_actions: self.column_actions,
                    coordinate_type,
                });

                Dataset::from_loader(loader)
            }
            DatasetSource::Manual => {
                let mut dataset = Dataset::new(coordinate_type);

                dataset.data = self
                    .points
                    .iter()
                    .map(|p| Datapoint {
                        point: p.clone(),
                        metadata: HashMap::new(),
                    })
                    .collect();

                Ok(dataset)
            }
            DatasetSource::None => bail!(DatasetBuilderError::NoDatasetSourceSet),
        }
    }
}

impl Default for DatasetBuilder {
    fn default() -> Self {
        Self {
            source: DatasetSource::default(),
            csv_delimiter: b',',
            csv_header: false,
            column_actions: Vec::new(),
            coordinate_type: None,
            points: Vec::new(),
        }
    }
}
