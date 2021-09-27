use std::cell::RefCell;
use std::f32::consts::PI;

use iced::{Container, Element, Length, Sandbox, Settings};
use num::complex::Complex32;
use num::traits::Pow;
use radar_lib::antenna::RectangularAperture;
use radar_lib::geodesic_polyhedron::generate_polyhedron;
use radar_lib::helper::{decibels, normalize_all};
use radar_lib::helper_traits::SphericalFunction;

use ndarray::{Array1, Axis, Slice};

use plotters::{
    coord::Shift,
    prelude::{LabelAreaPosition, LineSeries, PathElement},
    style::{AsRelative, Color, IntoFont, RelativeSize, BLACK, RED, WHITE},
};
use plotters_iced::{Chart, ChartBuilder, ChartWidget, DrawingArea, DrawingBackend};
use radar_lib::array_ext::Pad;
use radar_lib::radar::apply_matched_filter;
use radar_lib::series_chart::{ChartSet, SeriesChart};
use radar_lib::signal::fft::FFT;
use radar_lib::signal::fir::dolph_chebychev;
use radar_lib::signal::scalar::Scalar;
use radar_lib::signal::{
    convolve, fir, nsinc, qam_demodulate, qam_modulate, sinc, GaussianNoise, Pulse, PulseType,
    SampledDomain, Signal, Sinc, Sine,
};
use rand::prelude::StdRng;
use rand::SeedableRng;
use rustfft::FftPlanner;

struct State {
    chart: SeriesChart,
}

impl Sandbox for State {
    type Message = ();

    fn new() -> Self {
        let time = SampledDomain::new(0., 1., 250.);
        // let pulse = Pulse {
        //     clock: 0.0.into(),
        //     length: 1.,
        //     start_time: 0.,
        //     amplitude: 1.,
        //     sig_type: PulseType::Rectangle,
        // }
        // .generate_signal(&time);
        let pulse = fir::tukey(time.sample_count(), 0.3);
        // let time2 = SampledDomain::new(0., 100., 1000.);
        // let pulse2 = pulse.pad_back(0., time2.sample_count() - time.sample_count());
        // let modified = convolve::<f32>(&pulse.mapv(|x| x.re), &fir::tukey(pulse.len(), 0.8));
        let modulated = qam_modulate(pulse.mapv(|x| x.into()), time.clone(), 1e2.into());
        let demodulated = qam_demodulate(modulated, 1e2.into(), time.clone(), 1.);
        Self {
            chart: SeriesChart::new(
                time.fft_frequencies_plottable().into(),
                demodulated.fft_plottable().mapv(|x| x.norm()),
            ),
            // chart: SeriesChart::new(time.into(), demodulated.mapv(|x| x.re)),
        }
    }

    fn title(&self) -> String {
        "Split Chart Example".to_owned()
    }

    fn update(&mut self, _message: Self::Message) {}

    fn view(&mut self) -> Element<Self::Message> {
        let chart_view = ChartWidget::new(&mut self.chart)
            .width(Length::Fill)
            .height(Length::Fill);

        Container::new(chart_view)
            .width(Length::Fill)
            .height(Length::Fill)
            .padding(5)
            .center_x()
            .center_y()
            .into()
    }
}

fn main() {
    State::run(Settings {
        antialiasing: true,
        ..Settings::default()
    })
    .unwrap();
}
