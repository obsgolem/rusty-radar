use std::f32::consts::PI;

use ndarray::{linspace::linspace, Array1};
use num::complex::Complex32;

use super::{sampling_freq_to_len, Signal, Sinc};

/*
Generates the coefficients for FIR lowpass filter.

Given a time interval of a signal, the sampling frequency for that signal, and a stopband,
generates the coefficients for a filter of that length that will null out all frequency
components above the stopband. You may apply this to your original signal using convolution.

Example:
```rust
let filter = sinc_lowpass(dt, fs, b);
let filtered_signal = convolve(&signal, &filter);
```
*/
pub fn sinc_lowpass(interval: f32, sampling_freq: f32, stopband: f32) -> Array1<Complex32> {
    let len = sampling_freq_to_len(interval, sampling_freq);

    let window = hamming_window(len);

    // Normalization factor = 1/sampling_freq=sampling_interval
    let normalization = 1. / sampling_freq;

    let sinc = Sinc {
        bandwidth: stopband,
    };

    linspace(-interval / 2., interval / 2., len)
        .map(|t| Complex32::new(sinc.generate(t), 0.))
        .collect::<Array1<Complex32>>()
        * window
        * normalization
}

pub fn hamming_window(size: usize) -> Array1<f32> {
    const a0: f32 = 25. / 46.;
    const a1: f32 = 1. - a0;
    (0..size)
        .map(|n| a0 - a1 * (2. * PI * n as f32 / size as f32).cos())
        .collect()
}
