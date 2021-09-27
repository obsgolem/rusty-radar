/*!
Contains high and low fidelity algorithms for working with signals.

This library provides tools for working with continuous time signals through the [Signal] trait, together with a number of built in signal types.

It also provides a number of tools for working with discrete time signals. [SampledDomain] represents a time or frequency domain that has been sampled at a frequency.
[convolve] and friends allow you to apply fir filters. The [fir] module provides a number of windows for filter design, together with a low pass filter [fir::sinc_lowpass].
The [fft] module provides a trait which makes working with fourier transforms of [ndarray::Array1] a bit easier. The [iir] module will eventually provide tools for making and applying iir filters.

High fidelity functions for quadrature modulation and demodulation are available through [qam_modulate] and [qam_demodulate].
*/

use std::{
    borrow::BorrowMut,
    cell::RefCell,
    f32::consts::PI,
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Mul, MulAssign},
    sync::Arc,
};

use ndarray::{linspace::linspace, Array1, Zip};
use num::{complex::Complex32, traits::FloatConst, Float};
use rand::Rng;
use rand_distr::{Distribution, Normal};

use self::{fft::FFT, fir::sinc_lowpass, scalar::Scalar};

pub mod fft;
pub mod fir;
pub mod iir;
pub mod scalar;

fn sampling_freq_to_len(interval: f32, sampling_frequency: f32) -> usize {
    (interval * sampling_frequency).round() as usize
}

/// Performs the operation `sin(x)/x`
pub fn sinc<F: Float>(x: F) -> F {
    if x == F::zero() {
        F::one()
    } else {
        x.sin() / x
    }
}

/// Performs the operation `sin(πx)/πx`
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

/// A trait providing an abstract representation of a signal.
///
/// An implementor of this trait represents a continuous-time signal. The trait provides a uniform interface for sampling such signals.
/// Implementing this trait does not require determinism, see [GaussianNoise].
/// If your [Signal::generate] method has a large overhead which is needed only once for multiple samples, consider implementing [Signal::generate_signal] in addition to [Signal::generate].
pub trait Signal {
    type Valued: Scalar;
    fn generate(&self, t: f32) -> Self::Valued;

    fn generate_signal(&self, time: &SampledDomain) -> Array1<Self::Valued> {
        time.iter().map(|t| self.generate(t)).collect()
    }
}

/**
Represents a sampled time or frequency range.

This captures all the relevant properties of a range including the start and end of the range, the sampling frequency, the interval between samples, and the total number of samples.
The samples can be iterated over using [`SampledDomain::iter`] and can be converted to an array of samples using `into`.
*/
#[derive(Clone)]
pub struct SampledDomain {
    start: f32,
    end: f32,
    freq: f32,
    interval: f32,
    samples: usize,
}

impl SampledDomain {
    /// Constructs a time or frequency range using the given sampling frequency.
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

    /// Constructs a time or frequency range using the given number of evenly spaced samples.
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

    /// Constructs a time or frequency range using the given interval between samples and a given number of samples.
    pub fn from_sample_interval(start: f32, interval: f32, num: usize) -> SampledDomain {
        let freq = 1. / interval;
        let end = start + (num as f32 * interval);
        SampledDomain {
            start,
            end,
            freq,
            interval,
            samples: num,
        }
    }

    /// Returns the start time/frequency of the domain.
    pub fn start(&self) -> f32 {
        self.start
    }

    /// Returns the end time/frequency of the domain,
    pub fn end(&self) -> f32 {
        self.end
    }

    /// Returns the sampling frequency of the domain.
    pub fn freq(&self) -> f32 {
        self.freq
    }

    /// Returns the number of samples in the domain.
    pub fn sample_count(&self) -> usize {
        self.samples
    }

    /// The interval (in time or frequency) between individual samples.
    pub fn sample_interval(&self) -> f32 {
        self.interval
    }

    /// The interval between the start value and end value.
    pub fn range(&self) -> f32 {
        self.end - self.start
    }

    pub fn iter(&self) -> impl Iterator<Item = f32> {
        linspace(self.start, self.end, self.samples)
    }

    /// Converts the time domain to the frequency domain and vice versa. If you need to plot this you might want to consider [fft_frequencies_plottable] instead.
    pub fn fft_frequencies(&self) -> SampledDomain {
        Self::from_sample_interval(0., 1. / (self.range()), self.samples)
    }

    /// The fft is periodic as described at https://mathematica.stackexchange.com/a/33625. In order to plot this we need the frequencies arranged correctly.
    /// In order to use this, you will need to rotate the transformed signal as described in the link.
    pub fn fft_frequencies_plottable(&self) -> SampledDomain {
        let recip = 1. / (self.range());
        Self::from_sample_interval(
            -recip * ((self.samples / 2) as f32), //
            recip,
            self.samples,
        )
    }
}

impl From<SampledDomain> for Array1<f32> {
    fn from(time: SampledDomain) -> Self {
        time.iter().collect()
    }
}

/// Represents a sine wave of unit amplitude and a given frequency.
///
/// A sine wave can also have a phase shift given in radians. A wave of a given frequency can be constructed easily using `f32::into`.
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

/**
Represents a (potentially frequency-modulated) pulse.

A pulse can be either a rectangular pulse or a linear frequency sweep called a chirp.
If the clock on the pulse has a non-zero frequency, then it represents the carrier wave on which to modulate the baseband pulse signal.
This will be done by multiplying the baseband pulse by the carrier frequency.

This struct provides a lo-fidelity way of modeling Quadrature Amplitude Modulation (QAM).
By using this struct, you can abstract away the complex signal processing that needs to happen to modulate and demodulate a signal from the carrier frequency.
To modulate, simply set the clock to a non-zero frequency. To demodulate, simply set it back to zero. This gives better results than are physically realizable,
but should be sufficient for a lo-fidelity simulation. For higher fidelity, see [qam_modulate] and [qam_demodulate].
*/
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

/// Represents the type baseband signal of a [Pulse].
#[derive(Clone)]
pub enum PulseType {
    /// A rectangular wave, which is a constant 1 on the interval of the pulse.
    Rectangle,
    /// A chirp pulse, which starts at frequency 0 then increases in frequency to the given frequency.
    Chirp(f32),
}

/// Implements a random gaussian signal.
///
/// This signal is inherently non-deterministic, and calling [GaussianNoise::generate] twice with the same value
/// will almost certainly yield different results.
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

/// Represents a sinc wave/filter with a given bandwidth.
pub struct Sinc {
    pub bandwidth: f32,
}

impl Signal for Sinc {
    type Valued = f32;

    fn generate(&self, t: f32) -> Self::Valued {
        2. * self.bandwidth * nsinc(2. * self.bandwidth * t)
    }
}

/// A sawtooth wave.
///
/// This wave starts at a given amplitude, decreases to negative amplitude over a period, then jumps to the start.
#[derive(Clone)]
pub struct Sawtooth {
    pub period: f32,
    pub amplitude: f32,
}

impl Signal for Sawtooth {
    type Valued = f32;

    fn generate(&self, t: f32) -> Self::Valued {
        2. * self.amplitude * (t / self.period - (0.5 + t / self.period).floor())
    }
}

/// Ultra-slow naive convolution function
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

/// Convolve two real arrays using ffts.
pub fn convolve_f32(x: &Array1<f32>, y: &Array1<f32>) -> Array1<f32> {
    (x.clone().fft() * y.clone().fft())
        .ifft_plottable()
        .mapv(|x| x.re())
}

/// Convolve two complex arrays using ffts.
pub fn convolve_complex32(x: &Array1<Complex32>, y: &Array1<Complex32>) -> Array1<Complex32> {
    (x.clone().fft() * y.clone().fft()).ifft_plottable()
}

/**
Modulates a baseband signal onto a carrier wave retrieved from the clock.

The given time domain should be the time domain of the baseband signal.
This functions by multiplying the baseband signal by an exponential oscillating at the carrier frequency, then taking the real part of that signal.
This is useful because only real signals are physically realizable by e.g. an EM wave.

```rust
let time = SampledDomain::new(0., 1., 300.);
let clock: Sine = 100.0.into();
let signal = tukey(time.sample_count(), 0.3);
qam_modulate(signal.mapv(|x| x.into()), time, clock);
```
*/
pub fn qam_modulate(signal: Array1<Complex32>, time: SampledDomain, clock: Sine) -> Array1<f32> {
    assert!(signal.len() == time.sample_count(), "Samples in the signal being modulated must match the number of samples represented by the time domain.");
    (signal * clock.generate_signal(&time)).mapv(|x| x.re)
}

/**
Inverse of [qam_modulate].

In order to demodulate you need to remove the high frequency carrier signal and retreive the baseband waveform.
We do this by mixing the signal separately with the real and complex part of the original carrier wave.
If you look at this in the frequency domain then this will result in an extra copy of the signal appearing at `+-2f_0` where `f_0` is the carrier frequency.
We can apply a so called low-pass filter to remove all frequency components that are above the (one sided) bandwidth of the original complex signal.
In order for this to work well, the baseband signal should have bandwidth that is less than the carrier frequency, preferably significantly less.
Note that the bandwidth input on this function need only be a reasonable estimate of the actual bandwidth.

Let `f_0` be the clock frequency. Let `B` be the (one sided) bandwidth of the transmitted signal.
Let `B'` be the bandwidth of the low pass filter used for demodulation. Let `f_s` be the sampling frequency. Then in order for this function to work
the following relation must hold: `f_s > 2f_0+B+B'`. We check a simplified version of this assumption via assertion
*/
pub fn qam_demodulate(
    signal: Array1<f32>,
    clock: Sine,
    time: SampledDomain,
    bandwidth: f32,
) -> Array1<Complex32> {
    assert!(bandwidth < clock.freq, "Bandwidth is too large to disambiguate between the carrier frequency and the baseband signal.");
    assert!(
        time.freq() > 2. * clock.freq + bandwidth,
        "Sampling frequency is too small. The high frequency component of the \
    demodulated will alias to within the passband of the lowpass filter"
    );
    let dt = time.range();

    let demodulate_signal = clock.generate_signal(&time);

    // A lot of sources will conjugate y here, but this doesn't change the result due to the low pass filter.
    let pre_low_pass = Zip::from(&signal)
        .and(&demodulate_signal)
        .map_collect(|x, y| Complex32::new(2. * x * y.re, 2. * x * y.im));

    // From the note in the description we know we need `B'<f_s-2f_0-B`. We also want `B'>B` and `B'<2f_0-B`.
    // Any number satisfying these properties will suffice. Thus we will split the difference.
    let pass_bandwidth = if time.freq() - 2. * clock.freq - bandwidth < 2. * clock.freq - bandwidth
    {
        0.5 * time.freq() - clock.freq
    } else {
        clock.freq
    };

    convolve_complex32(
        &pre_low_pass,
        &sinc_lowpass(dt, signal.len() as f32 / dt, pass_bandwidth),
    )
}

mod test {
    use approx::assert_relative_eq;
    use ndarray::Axis;

    use crate::signal::{fir::tukey, qam_demodulate, qam_modulate, SampledDomain};

    use super::{Pulse, PulseType, Signal, Sine};

    #[test]
    fn fft_frequencies_involution() {
        let domain = SampledDomain::new(0., 20., 1.);
        let domain2 = domain.fft_frequencies().fft_frequencies();

        assert_relative_eq!(domain.start(), domain2.start());
        assert_relative_eq!(domain.end(), domain2.end());

        assert_relative_eq!(domain.freq(), domain2.freq());
    }

    #[test]
    fn modulate_demodulate_inverse() {
        let clock = Sine {
            freq: 100.0,
            phase: 0.0,
        };
        let time = SampledDomain::new(0., 1., 1000.);

        let pulse_unmodulated = tukey(time.sample_count(), 0.3);
        let demodulated = qam_demodulate(
            qam_modulate(
                pulse_unmodulated.mapv(|x| x.into()),
                time.clone(),
                clock.clone(),
            ),
            clock,
            time.clone(),
            1.,
        )
        .mapv(|x| x.re);

        // We slice out the edges of the signal where noise from the low pass filter is most likely to appear
        assert_relative_eq!(
            demodulated.slice_axis(Axis(0), (150..(1000 - 150)).into()),
            pulse_unmodulated.slice_axis(Axis(0), (150..(1000 - 150)).into()),
            max_relative = 1e-2
        );
    }
}
