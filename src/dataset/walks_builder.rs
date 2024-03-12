use anyhow::Context;
use thiserror::Error;
use time::format_description::parse_borrowed;
use time::macros::format_description;
use time::PrimitiveDateTime;

use crate::dataset::loader::CoordinateType;
use crate::dataset::point::Coordinates;
use crate::dataset::Dataset;
use crate::dp::DynamicProgramPool;
use crate::walk::Walk;
use crate::walker::Walker;

#[derive(Error, Debug)]
pub enum DatasetWalksBuilderError {
    #[error("a dataset must be provided")]
    NoDatasetSet,
    #[error("a dynamic program must be provided")]
    NoDynamicProgramSet,
    #[error("a walker must be provided")]
    NoWalkerSet,
    #[error("the number of time steps for the walks must be set or auto time steps must be used")]
    NoTimeStepsSet,
    #[error("the dataset must contain XY points for walk computation")]
    DatasetNotXY,
}

#[derive(Default, Debug, PartialEq)]
pub enum TimeStepsBy {
    Fixed(usize),
    TimeDifference(f64, &'static str),
    Distance(f64),
    #[default]
    None,
}

pub struct DatasetWalksBuilder<'a> {
    dataset: Option<&'a Dataset>,
    dp: Option<&'a DynamicProgramPool>,
    walker: Option<&'a Box<dyn Walker>>,
    from: usize,
    to: Option<usize>,
    count: usize,
    time_steps: TimeStepsBy,
    time_format: Option<&'static str>,
    auto_scale: bool,
    extra_steps: usize,
}

impl<'a> Default for DatasetWalksBuilder<'a> {
    fn default() -> Self {
        Self {
            dataset: None,
            dp: None,
            walker: None,
            from: 0,
            to: None,
            count: 1,
            time_steps: TimeStepsBy::None,
            time_format: None,
            auto_scale: false,
            extra_steps: 0,
        }
    }
}

impl<'a> DatasetWalksBuilder<'a> {
    pub fn new() -> Self {
        Self {
            ..Default::default()
        }
    }

    pub fn dataset(mut self, dataset: &'a Dataset) -> Self {
        self.dataset = Some(dataset);

        self
    }

    pub fn dp(mut self, dp: &'a DynamicProgramPool) -> Self {
        self.dp = Some(dp);

        self
    }

    pub fn walker(mut self, walker: &'a Box<dyn Walker>) -> Self {
        self.walker = Some(walker);

        self
    }

    pub fn from(mut self, from: usize) -> Self {
        self.from = from;

        self
    }

    pub fn to(mut self, to: usize) -> Self {
        self.to = Some(to);

        self
    }

    pub fn count(mut self, count: usize) -> Self {
        self.count = count;

        self
    }

    pub fn time_steps(mut self, time_steps: usize) -> Self {
        self.time_steps = TimeStepsBy::Fixed(time_steps);

        self
    }

    /// Automatically compute the number of time steps allowed for each walk.
    ///
    /// When set, the number of time steps allowed for each walk is automatically computed
    /// by comparing the timestamps of two points and calculating the time difference between
    /// two points. The time difference is then mapped to time steps using `time_step_len` which
    /// specifies the length of a time step in seconds. `metadata_key` specifies where
    /// timestamps are stored for each point.
    ///
    /// Use [`time_format()`] to set the format of the time string. If not set, the default format
    /// is used: `year-month-day hour:minute:second`.
    pub fn time_steps_by_time(mut self, time_step_len: f64, metadata_key: &'static str) -> Self {
        self.time_steps = TimeStepsBy::TimeDifference(time_step_len, metadata_key);

        self
    }

    /// Set the format of time strings in metadata used with automatic computation of time steps
    /// based on time difference.
    ///
    /// This only has an effect if used together with [`time_steps_by_time()`]. If not set, the
    /// default format is used: `year-month-day hour:minute:second`.
    pub fn time_format(mut self, format: &'static str) -> Self {
        self.time_format = Some(format);

        self
    }

    pub fn time_steps_by_dist(mut self, multiplier: f64) -> Self {
        self.time_steps = TimeStepsBy::Distance(multiplier);

        self
    }

    pub fn auto_scale(mut self) -> Self {
        self.auto_scale = true;

        self
    }

    pub fn set_auto_scale(mut self, auto_scale: bool) -> Self {
        self.auto_scale = auto_scale;

        self
    }

    pub fn extra_steps(mut self, extra_steps: usize) -> Self {
        self.extra_steps = extra_steps;

        self
    }

    pub fn build(self) -> anyhow::Result<Vec<Walk>> {
        let Some(dataset) = self.dataset else {
            return Err(DatasetWalksBuilderError::NoDatasetSet)?;
        };
        let Some(dp) = self.dp else {
            return Err(DatasetWalksBuilderError::NoDynamicProgramSet)?;
        };
        let Some(walker) = self.walker else {
            return Err(DatasetWalksBuilderError::NoWalkerSet)?;
        };

        if dataset.coordinate_type() != CoordinateType::XY {
            return Err(DatasetWalksBuilderError::DatasetNotXY)?;
        }

        if self.time_steps == TimeStepsBy::None {
            return Err(DatasetWalksBuilderError::NoTimeStepsSet)?;
        }

        let to = match self.to {
            Some(to) => to,
            None => dataset.len() - 1,
        };

        let format = match self.time_format {
            Some(format) => parse_borrowed::<2>(format).context("invalid time format string")?,
            None => format_description!("[year]-[month]-[day] [hour]:[minute]:[second]").to_vec(),
        };

        let mut walks = Vec::new();

        for i in self.from..to {
            let time_steps = match self.time_steps {
                TimeStepsBy::Fixed(time_steps) => time_steps,
                TimeStepsBy::TimeDifference(time_step_len, metadata_key) => {
                    let datetime1 = PrimitiveDateTime::parse(
                        dataset.get(i).unwrap().metadata.get(metadata_key).unwrap(),
                        &format,
                    )?;
                    let datetime2 = PrimitiveDateTime::parse(
                        dataset
                            .get(i + 1)
                            .unwrap()
                            .metadata
                            .get(metadata_key)
                            .unwrap(),
                        &format,
                    )?;

                    let diff = (datetime2 - datetime1).as_seconds_f64();

                    println!(
                        "Time difference: {}, time steps: {}",
                        diff,
                        diff / time_step_len
                    );

                    (diff / time_step_len) as usize
                }
                TimeStepsBy::Distance(multiplier) => {
                    let point1 = dataset.get(i).unwrap().clone().point;
                    let point2 = dataset.get(i + 1).unwrap().clone().point;

                    let (x1, y1): (i64, i64) = (point1.x(), point1.y());
                    let (x2, y2): (i64, i64) = (point2.x(), point2.y());

                    let dist = (x1 - x2).abs() + (y1 - y2).abs();

                    println!("Time steps: {}", (dist as f64 * multiplier) as usize);

                    (dist as f64 * multiplier) as usize
                }
                TimeStepsBy::None => {
                    unimplemented!("this should not happen because of the check above")
                }
            };

            for _ in 0..self.count {
                walks.push(
                    dataset
                        .rw_between(
                            dp,
                            walker,
                            i,
                            i + 1,
                            time_steps,
                            self.auto_scale,
                            self.extra_steps,
                        )
                        .context("could not generate walk")?,
                );
            }
        }

        Ok(walks)
    }
}
