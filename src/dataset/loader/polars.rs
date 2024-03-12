use std::collections::HashMap;

use anyhow::bail;
use polars::frame::DataFrame;

use crate::dataset::loader::{ColumnAction, CoordinateType, DatasetLoader, DatasetLoaderError};
use crate::dataset::point::{GCSPoint, Point, XYPoint};
use crate::dataset::Datapoint;

#[derive(Debug)]
pub struct PolarsLoaderOptions {
    pub df: DataFrame,
    pub column_actions: Vec<ColumnAction<String>>,
    pub coordinate_type: CoordinateType,
}

impl Default for PolarsLoaderOptions {
    fn default() -> Self {
        Self {
            df: DataFrame::empty(),
            column_actions: Vec::new(),
            coordinate_type: CoordinateType::default(),
        }
    }
}

pub struct PolarsLoader {
    options: PolarsLoaderOptions,
}

impl PolarsLoader {
    pub fn new(options: PolarsLoaderOptions) -> Self {
        Self { options }
    }
}

impl DatasetLoader for PolarsLoader {
    fn load(&self) -> anyhow::Result<Vec<Datapoint>> {
        if !self.options.column_actions.contains(&ColumnAction::KeepX) {
            bail!(DatasetLoaderError::NoXColumnSpecified);
        }
        if !self.options.column_actions.contains(&ColumnAction::KeepY) {
            bail!(DatasetLoaderError::NoYColumnSpecified);
        }

        let mut data = Vec::new();
        let cols = self.options.df.get_columns();

        for series in 0..self.options.df.iter().len() {
            let series = cols
                .iter()
                .map(|c| c.get(series).unwrap())
                .collect::<Vec<_>>();

            if series.len() != self.options.column_actions.len() {
                bail!(DatasetLoaderError::MoreColumnsThanActions);
            }

            let mut point = match self.options.coordinate_type {
                CoordinateType::GCS => Point::GCS(GCSPoint::default()),
                CoordinateType::XY => Point::XY(XYPoint::default()),
            };
            let mut metadata = HashMap::new();

            for (i, column) in series.iter().enumerate() {
                let column = column.to_string();

                match &self.options.column_actions[i] {
                    ColumnAction::KeepX => {
                        if let Point::GCS(point) = &mut point {
                            point.x = column.parse()?;
                        }
                    }
                    ColumnAction::KeepY => {
                        if let Point::GCS(point) = &mut point {
                            point.y = column.parse()?;
                        }
                    }
                    ColumnAction::KeepMetadata(key) => {
                        metadata.insert(key.clone(), column.to_string());
                    }
                    ColumnAction::Discard => (),
                }
            }

            let datapoint = Datapoint { point, metadata };

            data.push(datapoint);
        }

        Ok(data)
    }

    fn stream(&self) -> anyhow::Result<()> {
        todo!()
    }

    fn coordinate_type(&self) -> CoordinateType {
        self.options.coordinate_type
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use polars::df;
    use polars::prelude::NamedFrom;

    use crate::dataset::loader::polars::{PolarsLoader, PolarsLoaderOptions};
    use crate::dataset::loader::{ColumnAction, CoordinateType};
    use crate::dataset::point::{Point, XYPoint};
    use crate::dataset::{Datapoint, Dataset};

    #[test]
    fn test_polars_loader() {
        let df = df!(
            "agent_id" => &[1, 1, 2],
            "x" => &[10, 25, -17],
            "y" => &[5, 10, 28],
            "type" => &["a", "b", "a"],
        )
        .unwrap();

        let loader = PolarsLoader::new(PolarsLoaderOptions {
            df,
            column_actions: vec![
                ColumnAction::KeepMetadata("agent_id".into()),
                ColumnAction::KeepX,
                ColumnAction::KeepY,
                ColumnAction::Discard,
            ],
            coordinate_type: CoordinateType::XY,
        });

        let polars_dataset = Dataset::from_loader(loader).unwrap();

        let mut dataset = Dataset::new(CoordinateType::XY);

        dataset.push(Datapoint {
            point: Point::XY(XYPoint::from((10, 5))),
            metadata: HashMap::from([("agent_id".into(), "1".into())]),
        });
        dataset.push(Datapoint {
            point: Point::XY(XYPoint::from((25, 10))),
            metadata: HashMap::from([("agent_id".into(), "1".into())]),
        });
        dataset.push(Datapoint {
            point: Point::XY(XYPoint::from((-17, 28))),
            metadata: HashMap::from([("agent_id".into(), "2".into())]),
        });

        assert_eq!(dataset.data, polars_dataset.data);
    }
}
