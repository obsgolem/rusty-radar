use std::cell::RefCell;
use std::f32::consts::PI;

use iced::{Container, Element, Length, Sandbox, Settings};
use num::complex::Complex32;
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
use radar_lib::radar::apply_matched_filter;
use radar_lib::series_chart::{ChartSet, SeriesChart};
use radar_lib::signal::{
    nsinc, qam_demodulate, sinc, GaussianNoise, Pulse, PulseType, Signal, Sinc, Sine,
};
use rand::prelude::StdRng;
use rand::SeedableRng;
use rustfft::FftPlanner;

struct State {
    chart: ChartSet,
}

impl Sandbox for State {
    type Message = ();

    fn new() -> Self {
        let clock = Sine {
            freq: 1.0,
            phase: 0.0,
        };

        let pulse_unmodulated = Pulse {
            clock: 0.0.into(),
            length: 10., // Bandwidth 1/10
            start_time: 0.,
            amplitude: 1.,
            sig_type: PulseType::Rectangle,
        }
        .generate_signal(-5., 15., 100.)
        .signal();

        let pulse = Pulse {
            clock: clock.clone(),
            length: 10., // Bandwidth 1/10
            start_time: 0.,
            amplitude: 1.,
            sig_type: PulseType::Rectangle,
        }
        .generate_signal(-5., 15., 100.);

        let pulse_modulated = pulse.signal_ref().mapv(|x| x.re);

        let demodulated = qam_demodulate(pulse_modulated, clock, -5., 20., 0.1);
        // let freqs = time_to_frequency_plottable(&pulse.time());
        Self {
            chart: ChartSet {
                0: vec![
                    SeriesChart::new(pulse.time(), demodulated.mapv(|x| x.norm())),
                    // SeriesChart::new(pulse.time(), demodulated.1),
                    // SeriesChart::new(pulse.time(), demodulated.1.mapv(|x| x.re)),
                ],
            },
            // chart: SeriesChart::new(fft.freq(), fft.signal().mapv(|x| x.norm())),
            // chart: SeriesChart::new(pulse.time(), pulse.signal().mapv(|x| x.re)),
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
