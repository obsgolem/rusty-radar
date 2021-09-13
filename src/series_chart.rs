use ndarray::Array1;
use plotters::{
    prelude::{LabelAreaPosition, LineSeries},
    style::{AsRelative, IntoFont, Palette100, PaletteColor, RED},
};
use plotters_iced::{Chart, ChartBuilder, DrawingBackend};

pub struct SeriesChart {
    x: Array1<f32>,
    y: Array1<f32>,
    border_x: f32,
    border_y: f32,
}
impl SeriesChart {
    pub fn new(x: Array1<f32>, y: Array1<f32>) -> SeriesChart {
        SeriesChart {
            x,
            y,
            border_x: 0.1,
            border_y: 0.1,
        }
    }

    pub fn bounds(&self) -> (f32, f32, f32, f32) {
        let min_x = self
            .x
            .iter()
            .copied()
            .min_by(|x, y| x.total_cmp(y))
            .unwrap()
            .max(f32::MIN);
        let min_y = self
            .y
            .iter()
            .copied()
            .min_by(|x, y| x.total_cmp(y))
            .unwrap()
            .max(f32::MIN);

        let max_x = self
            .x
            .iter()
            .copied()
            .max_by(|x, y| x.total_cmp(y))
            .unwrap()
            .min(f32::MAX);
        let max_y = self
            .y
            .iter()
            .copied()
            .max_by(|x, y| x.total_cmp(y))
            .unwrap()
            .min(f32::MAX);
        let dist_x = (max_x - min_x).clamp(1e-6, f32::MAX);
        let dist_y = (max_y - min_y).clamp(1e-6, f32::MAX);

        let bottom_x = (min_x - dist_x * self.border_x / 2.).max(f32::MIN);
        let bottom_y = (min_y - dist_y * self.border_y / 2.).max(f32::MIN);
        let top_x = (max_x + dist_x * self.border_x / 2.).min(f32::MAX);
        let top_y = (max_y + dist_y * self.border_y / 2.).min(f32::MAX);

        (bottom_x, top_x, bottom_y, top_y)
    }
}

impl Chart<()> for SeriesChart {
    fn build_chart<DB: DrawingBackend>(&self, mut builder: ChartBuilder<DB>) {
        let (bottom_x, top_x, bottom_y, top_y) = self.bounds();

        let mut chart = builder
            .caption("y=x^2", ("sans-serif", 50).into_font())
            .set_label_area_size(LabelAreaPosition::Left, (5i32).percent_width())
            .set_label_area_size(LabelAreaPosition::Bottom, (10i32).percent_height())
            .build_cartesian_2d(bottom_x..top_x, bottom_y..top_y)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(LineSeries::new(
                self.x.iter().copied().zip(self.y.iter().copied()),
                &RED,
            ))
            .unwrap();
    }
}

pub struct ChartSet(pub Vec<SeriesChart>);

impl Chart<()> for ChartSet {
    fn build_chart<DB: DrawingBackend>(&self, mut builder: ChartBuilder<DB>) {
        let (mut bottom_x, mut top_x, mut bottom_y, mut top_y) =
            (f32::MAX, -f32::MAX, f32::MAX, -f32::MAX);
        for series in self.0.iter() {
            let bounds = series.bounds();
            bottom_x = bounds.0.min(bottom_x);
            top_x = bounds.1.max(top_x);
            bottom_y = bounds.2.min(bottom_y);
            top_y = bounds.3.max(top_y);
        }

        let mut chart = builder
            .caption("y=x^2", ("sans-serif", 50).into_font())
            .set_label_area_size(LabelAreaPosition::Left, (5i32).percent_width())
            .set_label_area_size(LabelAreaPosition::Bottom, (10i32).percent_height())
            .build_cartesian_2d(bottom_x..top_x, bottom_y..top_y)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        for (i, series) in self.0.iter().enumerate() {
            chart
                .draw_series(LineSeries::new(
                    series.x.iter().copied().zip(series.y.iter().copied()),
                    &PaletteColor::<Palette100>::pick(i),
                ))
                .unwrap();
        }
    }
}
