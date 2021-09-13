use std::{
    borrow::BorrowMut,
    cell::RefCell,
    f32::consts::PI,
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Mul, MulAssign},
    sync::Arc,
};

use ndarray::{linspace::linspace, Array1, Axis, Zip};
use num::{
    complex::Complex32,
    iter::Range,
    traits::{float::FloatCore, FloatConst},
    Float, Integer, ToPrimitive,
};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rustfft::{Fft, FftPlanner};

use self::{
    fir::{hamming_window, sinc_lowpass},
    scalar::Scalar,
};

pub mod fft;
pub mod fir;
pub mod scalar;

pub fn sampling_freq_to_len(interval: f32, sampling_frequency: f32) -> usize {
    (interval * sampling_frequency).round() as usize
}

pub fn sinc<F: Float>(x: F) -> F {
    if x == F::zero() {
        F::one()
    } else {
        x.sin() / x
    }
}

pub fn nsinc<F: Float + FloatConst>(x: F) -> F {
    let x = x * F::PI();
    if x == F::zero() {
        F::one()
    } else {
        x.sin() / x
    }
}

fn rectangle(t: f32, width: f32) -> f32 {
    if t >= -width / 2. && t <= width / 2. {
        1.
    } else {
        0.
    }
}

pub trait Signal {
    type Valued: Scalar;
    fn generate(&self, t: f32) -> Self::Valued;

    fn generate_signal(&self, time: &SampledDomain) -> Array1<Self::Valued> {
        time.iter().map(|t| self.generate(t)).collect()
    }
}

// Represents a time or frequency interval, sampled at a given rate or interval.
#[derive(Clone)]
pub struct SampledDomain {
    start: f32,
    end: f32,
    freq: f32,
    interval: f32,
    samples: usize,
}

impl SampledDomain {
    pub fn new(start: f32, end: f32, freq: f32) -> SampledDomain {
        assert!(start <= end);
        let len = sampling_freq_to_len(end - start, freq);
        SampledDomain {
            start,
            end,
            freq,
            interval: 1. / freq,
            samples: len,
        }
    }

    pub fn from_sample_count(start: f32, end: f32, num: usize) -> SampledDomain {
        assert!(start <= end);
        let freq = num as f32 / (end - start);
        SampledDomain {
            start,
            end,
            freq,
            interval: 1. / freq,
            samples: num,
        }
    }

    pub fn from_sample_interval(start: f32, interval: f32, num: usize) -> SampledDomain {
        let freq = 1. / interval;
        let end = num as f32 * interval;
        SampledDomain {
            start,
            end,
            freq,
            interval,
            samples: num,
        }
    }

    pub fn start(&self) -> f32 {
        self.start
    }

    pub fn end(&self) -> f32 {
        self.end
    }

    pub fn freq(&self) -> f32 {
        self.freq
    }

    pub fn sample_count(&self) -> usize {
        self.samples
    }

    pub fn sample_interval(&self) -> f32 {
        self.interval
    }

    pub fn range(&self) -> f32 {
        self.end - self.start
    }

    pub fn iter(&self) -> impl Iterator<Item = f32> {
        linspace(self.start, self.end, self.samples)
    }

    pub fn fft_frequencies(&self) -> SampledDomain {
        Self::from_sample_interval(0., 1. / (self.range()), self.samples)
    }

    // The fft is periodic as described at https://mathematica.stackexchange.com/a/33625. In order to plot this we need the frequencies arranged correctly.
    // In order to use this, you will need to rotate the transformed signal as described in the link.
    pub fn fft_frequencies_plottable(&self) -> SampledDomain {
        Self::from_sample_interval(
            -((self.samples / 2) as f32), //
            1. / (self.range()),
            self.samples,
        )
    }
}

impl From<SampledDomain> for Array1<f32> {
    fn from(time: SampledDomain) -> Self {
        time.iter().collect()
    }
}

#[derive(Clone)]
pub struct Sine {
    pub freq: f32,
    pub phase: f32,
}

impl Signal for Sine {
    type Valued = Complex32;

    fn generate(&self, t: f32) -> Self::Valued {
        const i: Complex32 = Complex32::new(0., 1.);

        (i * (2. * PI * self.freq * t + self.phase)).exp()
    }
}

impl From<f32> for Sine {
    fn from(x: f32) -> Self {
        Sine { freq: x, phase: 0. }
    }
}

#[derive(Clone)]
pub struct Pulse {
    pub clock: Sine,
    pub length: f32,
    pub start_time: f32,
    pub amplitude: f32,
    pub sig_type: PulseType,
}

impl Signal for Pulse {
    type Valued = Complex32;

    fn generate(&self, t: f32) -> Self::Valued {
        const i: Complex32 = Complex32::new(0., 1.);

        match self.sig_type {
            PulseType::Rectangle => {
                self.amplitude
                    * rectangle(t - self.start_time - self.length / 2., self.length)
                    * self.clock.generate(t)
            }
            PulseType::Chirp(chirp) => {
                // Chirps from 0 to chirp at baseband, or f to f+chirp at passband
                let chirp_slope = chirp / (2. * self.length);

                self.amplitude
                    * rectangle(t - self.start_time - self.length / 2., self.length)
                    * (2. * PI * i * t * chirp_slope * (t - self.start_time)).exp()
                    * self.clock.generate(t)
            }
        }
    }
}

impl Pulse {
    pub fn delay(&self, t: f32) -> Pulse {
        let mut out = self.clone();
        out.start_time += t;
        out.clock.phase -= 2. * PI * out.clock.freq * t;

        out
    }

    pub fn scale(&self, a: f32) -> Pulse {
        let mut out = self.clone();
        out.amplitude *= a;
        out
    }

    pub fn remove_modulation(&self) -> Pulse {
        let mut sig = self.clone();
        sig.clock = Sine {
            freq: 0.,
            phase: 0.,
        };
        sig
    }
}

#[derive(Clone)]
pub enum PulseType {
    Rectangle,
    // Chirp, argument is the chirp-to frequency.
    Chirp(f32),
}

// Implements a random gaussian signal. This signal is inherently non-deterministic, and calling generate twice with the same value
// will almost certainly yield different results.
pub struct GaussianNoise<T: Rng> {
    rng: RefCell<T>,
    distr: Normal<f32>,
}

impl<T: Rng> GaussianNoise<T> {
    pub fn new(sigma: f32, rng: T) -> GaussianNoise<T> {
        GaussianNoise {
            rng: RefCell::new(rng),
            distr: Normal::new(0., sigma).unwrap(),
        }
    }
}

impl<'a, T: Rng> Signal for GaussianNoise<T> {
    type Valued = Complex32;

    fn generate(&self, _: f32) -> Self::Valued {
        let rng = &mut *self.rng.borrow_mut();
        Complex32::new(self.distr.sample(rng), self.distr.sample(rng))
    }

    fn generate_signal(&self, time: &SampledDomain) -> Array1<Self::Valued> {
        let rng = &mut *self.rng.borrow_mut();
        time.iter()
            .map(|_| Complex32::new(self.distr.sample(rng), self.distr.sample(rng)))
            .collect()
    }
}

// Represents a sinc wave/filter with a given bandwidth.
pub struct Sinc {
    pub bandwidth: f32,
}

impl Signal for Sinc {
    type Valued = f32;

    fn generate(&self, t: f32) -> Self::Valued {
        2. * self.bandwidth * nsinc(2. * self.bandwidth * t)
    }
}

#[derive(Clone)]
struct Sawtooth {
    pub period: f32,
    pub amplitude: f32,
}

impl Signal for Sawtooth {
    type Valued = f32;

    fn generate(&self, t: f32) -> Self::Valued {
        2. * self.amplitude * (t / self.period - (0.5 + t / self.period).floor())
    }
}

// Ultra-slow naive convolution function
pub fn convolve<T: Scalar>(x: &Array1<T>, y: &Array1<T>) -> Array1<T> {
    let mut z = x.clone();

    for i in (0 as isize)..(x.len() as isize) {
        let mut acc = T::zero();
        let lim = (y.len() / 2) as isize;
        for j in -lim..lim {
            if i < j || i - j >= x.len() as isize {
                continue;
            }
            acc += x[(i - j) as usize] * y[(lim + j) as usize];
        }
        z[i as usize] = acc;
    }

    z
}

/*
This function is black magic signal processing. If you have an arbitrary complex signal then you can modulate it onto
a wave of high frequency by multiplying by a complex exponential oscillating at that frequency. You can then take
the real part of the result to get a signal that carries the full data of the original signal, but which is physically
realizable e.g. as a EM wave. At the recieve side you get a delayed, phase shifted, frequency shifted, and dampened
version of the original. You then need to remove the high frequency component and retreive the complex waveform in order
to do things like matched filtering and coherent integration. We do this by multiplying by the original complex exponential
again. This yields a signal with extra data at 2f (with bandwidth b) where f is the modulation frequency.
We can apply a so called low-pass filter to remove all frequency components that are above the bandwidth of the original complex signal.
The result is the complex signal with all the relevant delays and shifts applied.
*/
pub fn qam_demodulate(
    signal: Array1<f32>,
    clock: Sine,
    t_start: f32,
    dt: f32,
    bandwidth: f32,
) -> Array1<Complex32> {
    let demodulate_signal = linspace(t_start, t_start + dt, signal.len())
        .map(|t| clock.generate(t))
        .collect::<Array1<Complex32>>();

    let pre_low_pass = Zip::from(&signal)
        .and(&demodulate_signal)
        .map_collect(|x, y| Complex32::new(2. * x * y.re, -2. * x * y.im));

    // This needs to be somewhere between b and 2f-b
    let pass_bandwidth = clock.freq - bandwidth;

    convolve(
        &pre_low_pass,
        &sinc_lowpass(dt, signal.len() as f32 / dt, pass_bandwidth),
    )
}

mod test {
    use approx::assert_relative_eq;

    use crate::signal::{qam_demodulate, SampledDomain};

    use super::{Pulse, PulseType, Signal, Sine};

    #[test]
    fn fft_frequencies_involution() {
        let domain = SampledDomain::new(0., 20., 1.);
        let domain2 = domain.fft_frequencies().fft_frequencies();

        assert_relative_eq!(domain.start(), domain2.start());

        assert_relative_eq!(domain.end(), domain2.end());

        assert_relative_eq!(domain.freq(), domain2.freq());
    }

    // #[test]
    // fn modulate_demodulate_inverse() {
    //     let clock = Sine {
    //         freq: 1.0,
    //         phase: 0.0,
    //     };

    //     let pulse_unmodulated = Pulse {
    //         clock: 0.0.into(),
    //         length: 10., // Bandwidth 1/10
    //         start_time: 0.,
    //         amplitude: 1.,
    //         sig_type: PulseType::Rectangle,
    //     }
    //     .generate_signal(-5., 15., 100.)
    //     .signal();

    //     let pulse = Pulse {
    //         clock: clock.clone(),
    //         length: 10., // Bandwidth 1/10
    //         start_time: 0.,
    //         amplitude: 1.,
    //         sig_type: PulseType::Rectangle,
    //     }
    //     .generate_signal(-5., 15., 100.);

    //     let pulse_modulated = pulse.signal_ref().mapv(|x| x.re);

    //     let demodulated = qam_demodulate(pulse_modulated, clock, -5., 20., 0.1);

    //     assert_relative_eq!(demodulated, pulse_unmodulated);
    // }
}
