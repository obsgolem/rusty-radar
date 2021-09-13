use std::sync::Arc;

use ndarray::Array1;
use num::complex::Complex32;
use rustfft::FftPlanner;

pub trait FFT {
    fn fft_planned(self, plan: &Arc<dyn rustfft::Fft<f32>>) -> Array1<Complex32>;
    fn fft(self) -> Array1<Complex32>;
    fn fft_plottable(self) -> Array1<Complex32>;

    fn ifft(self) -> Array1<Complex32>;
}

impl FFT for Array1<f32> {
    fn fft(self) -> Array1<Complex32> {
        self.mapv(|x| Complex32::new(x, 0.)).fft()
    }

    fn fft_plottable(self) -> Array1<Complex32> {
        self.mapv(|x| Complex32::new(x, 0.)).fft_plottable()
    }

    fn ifft(self) -> Array1<Complex32> {
        self.mapv(|x| Complex32::new(x, 0.)).ifft()
    }

    fn fft_planned(self, plan: &Arc<dyn rustfft::Fft<f32>>) -> Array1<Complex32> {
        self.mapv(|x| Complex32::new(x, 0.)).fft_planned(plan)
    }
}

impl FFT for Array1<Complex32> {
    fn fft_planned(mut self, plan: &Arc<dyn rustfft::Fft<f32>>) -> Array1<Complex32> {
        plan.process(self.as_slice_mut().unwrap());

        self
    }

    fn fft(self) -> Array1<Complex32> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.len());

        self.fft_planned(&fft)
    }

    fn fft_plottable(self) -> Array1<Complex32> {
        let mut out = self.fft();
        let len = out.len();
        out.as_slice_mut().unwrap().rotate_right((len + 1) / 2);
        out
    }

    fn ifft(self) -> Array1<Complex32> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_inverse(self.len());

        let N = self.len() as f32;

        self.fft_planned(&fft) / N
    }
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;

    use crate::signal::{fft::FFT, Pulse, PulseType, SampledDomain, Signal};

    #[test]
    fn ifft_commutes_with_fft() {
        let pulse = Pulse {
            clock: 0.0.into(),
            length: 10.,
            start_time: 0.,
            amplitude: 1.,
            sig_type: PulseType::Chirp(1.),
        }
        .generate_signal(&SampledDomain::new(0., 10., 1.));

        {
            let transformed = pulse.clone().fft().ifft();
            assert_relative_eq!(pulse, transformed, epsilon = 2. * f32::EPSILON);
        }

        {
            let transformed = pulse.clone().ifft().fft();
            assert_relative_eq!(pulse, transformed, epsilon = 2. * f32::EPSILON);
        }
    }
}
