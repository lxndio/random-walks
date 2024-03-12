//! Provides the basic data structure for a [`Walk`](Walk).
//!
//! A [`Walk`](Walk) can be created manually by specifying all of its points or generated using a
//! [`DynamicProgram`](crate::dp::DynamicProgramPool). See the [`dp`](crate::dp) module for more
//! information on how to generate random walks.
//!
//! The [`Walk`](Walk) structure also provides different useful functions for manipulating and
//! reviewing walks. If the `plotting` feature is enabled, walks can also be plotted to an
//! image file.

use std::collections::HashSet;
use std::ops::{Index, Range};

use anyhow::bail;
use geo::{line_string, Coord, FrechetDistance, LineString};
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::drawing::IntoDrawingArea;
use plotters::element::{Circle, EmptyElement, Text};
use plotters::prelude::{IntoFont, LineSeries, PointSeries, RGBColor, BLACK, WHITE};
use rand::Rng;

use crate::dataset::point::XYPoint;

/// A random walk consisting of multiple points.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct Walk(pub Vec<XYPoint>);

impl Walk {
    // Returns the number of steps in the walk.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    // Returns whether the walk contains any steps.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<XYPoint> {
        self.0.iter()
    }

    /// Computes the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between
    /// two random walks.
    ///
    /// ```
    /// # use randomwalks_lib::walker::Walk;
    /// # use randomwalks_lib::xy;
    /// #
    /// let walk1 = Walk(vec![xy!(0, 0), xy!(2, 2), xy!(5, 5)]);
    /// let walk2 = Walk(vec![xy!(0, 0), xy!(3, 3), xy!(6, 6)]);
    ///
    /// let frechet = walk1.frechet_distance(&walk2);
    /// ```
    pub fn frechet_distance(&self, other: &Walk) -> f64 {
        let self_line = LineString::from(self);
        let other_line = LineString::from(other);

        self_line.frechet_distance(&other_line)
    }

    /// Computes how much a random walk deviates from the straight line between the start and
    /// end point.
    pub fn directness_deviation(&self) -> f64 {
        let self_line = LineString::from(self);
        let other_line = line_string![
            (x: self.0.first().unwrap().x as f64, y: self.0.first().unwrap().y as f64),
            (x: self.0.last().unwrap().x as f64, y: self.0.last().unwrap().y as f64),
        ];

        self_line.frechet_distance(&other_line)
    }

    /// Translates all points of a walk.
    ///
    /// ```
    /// # use randomwalks_lib::walker::Walk;
    /// # use randomwalks_lib::dataset::point::XYPoint;
    /// # use randomwalks_lib::xy;
    /// #
    /// let walk1 = Walk(vec![xy!(0, 0), xy!(2, 3), xy!(7, 5)]).translate(xy!(5, 1));
    /// let walk2 = Walk(vec![xy!(5, 1), xy!(7, 4), xy!(12, 6)]);
    ///
    /// assert_eq!(walk1, walk2);
    /// ```
    pub fn translate(&self, by: XYPoint) -> Walk {
        Walk(
            self.0
                .iter()
                .map(|p| (p.x + by.x, p.y + by.y).into())
                .collect(),
        )
    }

    /// Scales all points of a walk.
    ///
    /// ```
    /// # use randomwalks_lib::walker::Walk;
    /// # use randomwalks_lib::dataset::point::XYPoint;
    /// # use randomwalks_lib::xy;
    /// #
    /// let walk1 = Walk(vec![xy!(0, 0), xy!(2, 3), xy!(7, 5)]).scale(xy!(2, 1));
    /// let walk2 = Walk(vec![xy!(0, 0), xy!(4, 3), xy!(14, 5)]);
    ///
    /// assert_eq!(walk1, walk2);
    /// ```
    pub fn scale(&self, by: XYPoint) -> Walk {
        Walk(
            self.0
                .iter()
                .map(|p| (p.x * by.x, p.y * by.y).into())
                .collect(),
        )
    }

    /// Rotates all points of a walk around the origin.
    ///
    /// ```
    /// # use randomwalks_lib::walker::Walk;
    /// # use randomwalks_lib::dataset::point::XYPoint;
    /// # use randomwalks_lib::xy;
    /// #
    /// let walk1 = Walk(vec![xy!(0, 0), xy!(2, 3), xy!(7, 5)]).rotate(90.0);
    /// let walk2 = Walk(vec![xy!(0, 0), xy!(-3, 2), xy!(-5, 7)]);
    ///
    /// assert_eq!(walk1, walk2);
    /// ```
    pub fn rotate(&self, degrees: f64) -> Walk {
        let rad = degrees.to_radians();

        Walk(
            self.0
                .iter()
                .map(|p| {
                    (
                        (p.x as f64 * rad.cos() - p.y as f64 * rad.sin()) as i64,
                        (p.y as f64 * rad.cos() + p.x as f64 * rad.sin()) as i64,
                    )
                        .into()
                })
                .collect(),
        )
    }

    /// Plots a walk and saves the resulting image to a .png file.
    ///
    /// ```
    /// # use randomwalks_lib::walker::Walk;
    /// # use randomwalks_lib::xy;
    /// #
    /// let walk = Walk(vec![xy!(0, 0), xy!(2, 3), xy!(7, 5)]);
    ///
    /// walk.plot("walk.png")?;
    /// ```
    #[cfg(feature = "plotting")]
    pub fn plot<S: Into<String>>(&self, filename: S) -> anyhow::Result<()> {
        if self.0.is_empty() {
            bail!("Cannot plot empty walk");
        }

        let filename = filename.into();

        // Initialize plot

        let (coordinate_range_x, coordinate_range_y) = point_range(&[self.clone()]);

        let root = BitMapBackend::new(&filename, (1000, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root.margin(10, 10, 10, 10);

        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(20)
            .y_label_area_size(20)
            .build_cartesian_2d(coordinate_range_x, coordinate_range_y)?;

        chart.configure_mesh().draw()?;

        // Draw walk

        let walk: Vec<(i64, i64)> = self.0.iter().map(|x| (*x).into()).collect();

        chart.draw_series(LineSeries::new(walk.to_vec(), &BLACK))?;

        // Draw start and end point

        chart.draw_series(PointSeries::of_element(
            vec![*walk.first().unwrap(), *walk.last().unwrap()],
            5,
            &BLACK,
            &|c, s, st| {
                EmptyElement::at(c)
                    + Circle::new((0, 0), s, st.filled())
                    + Text::new(format!("{:?}", c), (10, 0), ("sans-serif", 10).into_font())
            },
        ))?;

        Ok(())
    }

    /// Plots multiple walks together and saves the resulting image to a .png file.
    ///
    /// ```
    /// # use randomwalks_lib::walker::Walk;
    /// # use randomwalks_lib::xy;
    /// #
    /// let walk1 = Walk(vec![xy!(0, 0), xy!(2, 3), xy!(7, 5)]);
    /// let walk2 = Walk(vec![xy!(0, 0), xy!(5, 5), xy!(7, 8)]);
    /// let walks = vec![walk1, walk2];
    ///
    /// Walk::plot_multiple(&walks, "walks.png")?;
    /// ```
    #[cfg(feature = "plotting")]
    pub fn plot_multiple<S: Into<String>>(walks: &[Walk], filename: S) -> anyhow::Result<()> {
        let filename = filename.into();

        // Initialize plot

        let (coordinate_range_x, coordinate_range_y) = point_range(walks);

        let root = BitMapBackend::new(&filename, (1000, 1000)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root.margin(10, 10, 10, 10);

        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(20)
            .y_label_area_size(20)
            .build_cartesian_2d(coordinate_range_x, coordinate_range_y)?;

        chart.configure_mesh().draw()?;

        // Draw walks

        let walks: Vec<Vec<(i64, i64)>> = walks
            .iter()
            .map(|w| w.iter().map(|p| (p.x, p.y)).collect())
            .collect();

        let mut rng = rand::thread_rng();

        for walk in walks.iter() {
            chart.draw_series(LineSeries::new(
                walk.clone(),
                RGBColor(
                    rng.gen_range(30..220),
                    rng.gen_range(30..220),
                    rng.gen_range(30..220),
                ),
            ))?;
        }

        // Find unique start and end points

        let mut se_points = HashSet::new();

        for walk in walks.iter() {
            se_points.insert((
                walk.first().copied().unwrap(),
                walk.last().copied().unwrap(),
            ));
        }

        // Draw start and end points

        for (start, end) in se_points {
            chart.draw_series(PointSeries::of_element(
                vec![start, end],
                5,
                &BLACK,
                &|c, s, st| {
                    EmptyElement::at(c)
                        + Circle::new((0, 0), s, st.filled())
                        + Text::new(format!("{:?}", c), (10, 0), ("sans-serif", 10).into_font())
                },
            ))?;
        }

        Ok(())
    }
}

#[cfg(feature = "plotting")]
fn point_range(walks: &[Walk]) -> (Range<i64>, Range<i64>) {
    // Compute size of plotting area

    let points: Vec<_> = walks.iter().flat_map(|x| &x.0).copied().collect();

    let xs: Vec<i64> = points.iter().map(|p| p.x).collect();
    let ys: Vec<i64> = points.iter().map(|p| p.y).collect();

    let x_range = (*xs.iter().min().unwrap(), *xs.iter().max().unwrap());
    let y_range = (*ys.iter().min().unwrap(), *ys.iter().max().unwrap());

    let coordinate_range_x = x_range.0 - 5..x_range.1 + 5;
    let coordinate_range_y = y_range.1 + 5..y_range.0 - 5;

    (coordinate_range_x, coordinate_range_y)
}

impl From<Vec<XYPoint>> for Walk {
    fn from(value: Vec<XYPoint>) -> Self {
        Self(value)
    }
}

impl From<Walk> for Vec<XYPoint> {
    fn from(value: Walk) -> Self {
        value.0
    }
}

impl From<&Walk> for LineString<f64> {
    fn from(value: &Walk) -> Self {
        Self(
            value
                .0
                .iter()
                .map(|p| (p.x as f64, p.y as f64))
                .map(Coord::from)
                .collect(),
        )
    }
}

impl FromIterator<XYPoint> for Walk {
    fn from_iter<T: IntoIterator<Item = XYPoint>>(iter: T) -> Self {
        let mut c = Vec::new();

        for i in iter {
            c.push(i);
        }

        Self(c)
    }
}

impl Index<usize> for Walk {
    type Output = XYPoint;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::point::XYPoint;
    use crate::walk::Walk;
    use crate::xy;

    #[test]
    fn test_walk_translate() {
        let walk1 = Walk(vec![xy!(0, 0), xy!(2, 3), xy!(7, 5)]).translate(xy!(5, 1));
        let walk2 = Walk(vec![xy!(5, 1), xy!(7, 4), xy!(12, 6)]);

        assert_eq!(walk1, walk2);
    }

    #[test]
    fn test_walk_scale() {
        let walk1 = Walk(vec![xy!(0, 0), xy!(2, 3), xy!(7, 5)]).scale(xy!(2, 1));
        let walk2 = Walk(vec![xy!(0, 0), xy!(4, 3), xy!(14, 5)]);

        assert_eq!(walk1, walk2);
    }

    #[test]
    fn test_walk_rotate() {
        let walk1 = Walk(vec![xy!(0, 0), xy!(2, 3), xy!(7, 5)]).rotate(90.0);
        let walk2 = Walk(vec![xy!(0, 0), xy!(-3, 2), xy!(-5, 7)]);

        assert_eq!(walk1, walk2);
    }
}
