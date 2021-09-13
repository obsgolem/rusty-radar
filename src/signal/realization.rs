use std::sync::Arc;

use ndarray::{Array1, Axis};
use num::complex::Complex32;
use num::ToPrimitive;
use rustfft::{Fft, FftPlanner};

use super::fft;
use super::scalar::Scalar;

#[derive(Clone)]
pub struct SignalRealization<T: Scalar> {
    // Independent variable, usually time or frequency.
    // Assumed to be evenly spaced. Must be same length as signal.
    pub(crate) ind: Array1<f32>,
    // The actual value of the signal at the corresponding value of the independent variable.
    // Must be same length as ind
    pub(crate) signal: Array1<T>,
}

impl<T: Scalar> SignalRealization<T> {
    pub fn time(&self) -> Array1<f32> {
        return self.ind.clone();
    }

    pub fn freq(&self) -> Array1<f32> {
        return self.ind.clone();
    }

    pub fn independent_var(&self) -> Array1<f32> {
        return self.ind.clone();
    }

    pub fn signal(&self) -> Array1<T> {
        return self.signal.clone();
    }

    pub fn signal_mut_ref(&mut self) -> &mut Array1<T> {
        &mut self.signal
    }

    pub fn signal_ref(&self) -> &Array1<T> {
        &self.signal
    }

    pub fn signal_energy(&self) -> f32 {
        let mut zip_sig = self.ind.iter().copied().zip(self.signal.iter().copied());
        let last = zip_sig.next().unwrap();

        let mut acc = 0.0f32;
        for sig in zip_sig {
            let delta = sig.0 - last.0;
            acc += delta * last.1.norm_sqr().to_f32().unwrap();
        }

        acc
    }

    pub fn signal_power(&self) -> f32 {
        let t0 = *self.ind.first().unwrap();
        let t1 = *self.ind.last().unwrap();
        self.signal_energy() / (t1 - t0)
    }

    pub fn norm(&self) -> SignalRealization<T::CompType> {
        SignalRealization {
            ind: self.ind.clone(),
            signal: self.signal.mapv(|x| x.norm()),
        }
    }

    // Computes the coefficients of a filter matched to this signal.
    // The matched filter is given by the conjugated and time reversed
    // input signal.
    pub fn matched_filter_coefficients(&self) -> Array1<T> {
        let mut sig = self.signal.clone();
        sig.invert_axis(Axis(0));
        sig.mapv_inplace(|x| x.conj());

        sig
    }

    // Returns the index of first time >= the input time.
    pub fn time_index(&self, time: f32) -> Option<usize> {
        self.ind
            .iter()
            .copied()
            .enumerate()
            .find(|x| x.1 >= time)
            .map(|x| x.0)
    }

    pub fn add_from_index(
        &self,
        other: &SignalRealization<T>,
        index: usize,
        other_index: usize,
    ) -> SignalRealization<T> {
        let mut out = self.clone();
        let size = if other.signal.len() - other_index < self.signal.len() - index {
            other.signal.len() - other_index
        } else {
            self.signal.len() - index
        };
        let mut signal_slice = out
            .signal
            .slice_axis_mut(Axis(0), ndarray::Slice::from(index..index + size));
        let other_slice = other.signal.slice_axis(
            Axis(0),
            ndarray::Slice::from(other_index..other_index + size),
        );
        signal_slice += &other_slice;

        out
    }

    pub fn add_from_index_inplace(
        &mut self,
        other: &SignalRealization<T>,
        index: usize,
        other_index: usize,
    ) {
        let size = if other.signal.len() - other_index < self.signal.len() - index {
            other.signal.len() - other_index
        } else {
            self.signal.len() - index
        };
        let mut signal_slice = self
            .signal
            .slice_axis_mut(Axis(0), ndarray::Slice::from(index..index + size));
        let other_slice = other.signal.slice_axis(
            Axis(0),
            ndarray::Slice::from(other_index..other_index + size),
        );
        signal_slice += &other_slice;
    }
}

impl SignalRealization<Complex32> {
    pub fn fft_frequencies(&self) -> Array1<f32> {
        fft::time_to_frequency(&self.ind)
    }

    pub fn fft_frequencies_neg(&self) -> Array1<f32> {
        fft::time_to_frequency_neg(&self.ind)
    }

    pub fn fft_frequencies_shifted(&self) -> Array1<f32> {
        fft::time_to_frequency_plottable(&self.ind)
    }

    pub fn ifft_times(&self) -> Array1<f32> {
        fft::freqency_to_time(&self.ind)
    }

    // pub fn ifft_times_neg(&self) -> Array1<f32> {
    //     time_to_frequency_neg(&self.ind)
    // }

    // pub fn ifft_times_shifted(&self) -> Array1<f32> {
    //     time_to_frequency_plottable(&self.ind)
    // }

    pub fn fft_inplace(&mut self, plan: &Arc<dyn Fft<f32>>) {
        self.ind = self.fft_frequencies();
        plan.process(self.signal.as_slice_mut().unwrap());
    }

    pub fn fft(&self) -> SignalRealization<Complex32> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.ind.len());

        self.fft_planned(&fft)
    }

    pub fn fft_planned(&self, plan: &Arc<dyn Fft<f32>>) -> SignalRealization<Complex32> {
        let freq = self.fft_frequencies();
        let signal = self.signal.clone();

        let mut out = SignalRealization { ind: freq, signal };
        plan.process(out.signal.as_slice_mut().unwrap());

        out
    }

    // The FFT is in reverse order from what you might expect when looking at the plain old fourier transform.
    // To get the correct order, we do a rotate, as described here: https://mathematica.stackexchange.com/a/33625
    pub fn fft_plottable(&self) -> SignalRealization<Complex32> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(self.ind.len());

        let freq = self.fft_frequencies_shifted();
        let signal = self.signal.clone();

        let mut out = SignalRealization { ind: freq, signal };
        fft.process(out.signal.as_slice_mut().unwrap());

        out.signal
            .as_slice_mut()
            .unwrap()
            .rotate_right((out.ind.len() + 1) / 2);
        out
    }

    pub fn ifft_planned(&self, plan: &Arc<dyn Fft<f32>>) -> SignalRealization<Complex32> {
        let time = self.ifft_times();
        let signal = self.signal.clone();

        let mut out = SignalRealization { ind: time, signal };
        plan.process(out.signal.as_slice_mut().unwrap());
        let N = out.signal.len() as f32;
        out.signal.mapv_inplace(|x| x / N);
        out
    }

    pub fn ifft(&self) -> SignalRealization<Complex32> {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_inverse(self.ind.len());

        self.ifft_planned(&fft)
    }
}

impl SignalRealization<f32> {
    pub fn decibels(&self) -> SignalRealization<f32> {
        SignalRealization {
            ind: self.ind.clone(),
            signal: self.signal.mapv(|x| 10. * x.log10()),
        }
    }

    pub fn to_complex(&self) -> SignalRealization<Complex32> {
        SignalRealization {
            ind: self.ind.clone(),
            signal: self.signal.mapv(|x| Complex32::new(x, 0.)),
        }
    }
}
