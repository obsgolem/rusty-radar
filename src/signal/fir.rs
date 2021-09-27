use std::f32::consts::PI;

use ndarray::{linspace::linspace, Array1};
use num::complex::Complex32;

use crate::signal::fft::FFT;

use super::{sampling_freq_to_len, Signal, Sinc};

/**
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

/**
This version of the hamming window uses 25/46 rather than the rounded 0.54. See equation (134) in [1].

[1] Armin Doerry, "Catalog of Window Taper Functions for Sidelobe Control", 2017.
           https://www.researchgate.net/profile/Armin_Doerry/publication/316281181_Catalog_of_Window_Taper_Functions_for_Sidelobe_Control/links/58f92cb2a6fdccb121c9d54d/Catalog-of-Window-Taper-Functions-for-Sidelobe-Control.pdf
*/
pub fn hamming_window(size: usize) -> Array1<f32> {
    const a0: f32 = 25. / 46.;
    const a1: f32 = 1. - a0;
    linspace(-PI, PI, size).map(|x| a0 + a1 * x.cos()).collect()
}

/// Returns the fft of [dolph_chebychev].
/// This window is usually defined as the ifft of this function, so both are made available.
/// Note that this function is
/// 1) Unnormalized
/// 2) Probably not what you want anyways. The window is the real part of the ifft of this function, which messes with the frequencies a bit.
pub fn dolph_chebychev_fft(size: usize, α: f32) -> Array1<Complex32> {
    assert!(size > 1);

    const i: Complex32 = Complex32::new(0., 1.);

    let N = size as f32;

    let denom = 10.0f32.powf(α);
    let β = (denom.acosh() / (N - 1.)).cosh();

    fn T(n: usize, x: f32) -> f32 {
        let n1 = n as f32;
        if x >= -1. && x <= 1. {
            (n1 * x.acos()).cos()
        } else if x > 1. {
            (n1 * x.acosh()).cosh()
        } else {
            (-1f32).powi(n as i32) * (n1 * (-x).acosh()).cosh()
        }
    }

    if size % 2 == 1 {
        // Odd length, even order
        (0..size)
            .map(|k| {
                let x = k as f32;
                // Note that we omit the denom normalization factor in favor of normalizing after the fft.
                (T(size - 1, β * (PI * x / N).cos())).into()
            })
            .collect()
    } else {
        // Even length, odd order. In order to preserve the symmetry in the time domain we need to shift time by half a sample.
        // Doing this is equivalent to multiplying by an exponential factor in frequency.
        (0..size)
            .map(|k| {
                let x = k as f32;
                T(size - 1, β * (PI * x / N).cos()) * (x / N * PI * i).exp()
            })
            .collect()
    }
}

/**
Dolph-Chebychev is window which is optimal for a given sidelobe height.
The parameter α controls the height of the sidelobes. In decibels,
the sidelobes will have height -20α. For more information on this window see
https://ccrma.stanford.edu/~jos/sasp/Dolph_Chebyshev_Window.html
The information there is extremely unhelpful for actually implementing the window.
For a source with actual implementation details, see
https://github.com/scipy/scipy/blob/v1.7.1/scipy/signal/windows/windows.py#L1350-L1473
*/
pub fn dolph_chebychev(size: usize, α: f32) -> Array1<f32> {
    // We use the fft rather than the ifft here in order to avoid normalization. We instead normalize by the first element of the array.
    let mut out = dolph_chebychev_fft(size, α).fft().mapv(|x| x.re);
    out /= out[0];
    out.as_slice_mut().unwrap().rotate_right((size - 1) / 2);
    out
}

/**
The taylor window requires a set of coefficients which depend solely on the sidelobe levels and number of untapered sidelobes.
This function generates those coeffient so that they can be used in multiple windows of differing length. For more information, see [taylor]
*/
pub fn taylor_coefficients(η: f32, num_const_sidelobes: usize) -> Array1<f32> {
    let n_bar = num_const_sidelobes as f32;
    let A = η.acosh() / PI;
    let σ_sq = n_bar.powi(2) / (A * A + (n_bar - 0.5).powi(2));
    (1..num_const_sidelobes)
        .map(|i| -> f32 {
            let m = i as f32;
            let denom: f32 = (1..num_const_sidelobes)
                .filter(|j| *j != i)
                .map(|j| {
                    let n = j as f32;
                    1. - (m * m / (n * n))
                })
                .product();

            let num: f32 = (1..num_const_sidelobes)
                .map(|j| {
                    let n = j as f32;
                    1. - (m * m) / (σ_sq) / (A * A + (n - 0.5).powi(2))
                })
                .product();

            // (-1)^i with extra steps due to powi not taking a usize.
            let s = (2 * (i % 2)) as f32 - 1.;

            0.5 * s * num / denom
        })
        .collect()
}

/**
This function takes the output from [taylor_coefficients] and produces a window of the given size. See [taylor] for more information.
*/
pub fn taylor_from_coefficients(size: usize, coefficients: Array1<f32>) -> Array1<f32> {
    let num_const_sidelobes = coefficients.len() + 1;
    let M = size as f32;
    let Fm = coefficients;
    let W = |x: f32| -> f32 {
        let n = x as f32;
        1. + 2.
            * (1..num_const_sidelobes)
                .map(|i| {
                    let m = i as f32;
                    Fm[i - 1] * (2. * PI * m * (n - M / 2. + 0.5) / M).cos()
                })
                .sum::<f32>()
    };
    (0..size).map(|x| W(x as f32)).collect::<Array1<f32>>() / W((M - 1.) / 2.)
}

/**
Generates a Taylor window. This window is a variant of the dolph-chebychev window which provides a configurable taper on the sidelobes.
Unlike most implementations, this takes the sidelobe level η in linear space. To convert a decibel sidelobe level to linear use η=10^(x/20).
The second third parameter to this function, num_const_sidelobes, describes the number of sidelobes that should not be tapered. After going n
sidelobes out, every sidelobe after will have the taper applied.

[1] Armin Doerry, "Catalog of Window Taper Functions for Sidelobe Control", 2017.
           https://www.researchgate.net/profile/Armin_Doerry/publication/316281181_Catalog_of_Window_Taper_Functions_for_Sidelobe_Control/links/58f92cb2a6fdccb121c9d54d/Catalog-of-Window-Taper-Functions-for-Sidelobe-Control.pdf
[2] https://github.com/scipy/scipy/blob/97ea4e506c7a4c6fdd144c09e00522134dd64c94/scipy/signal/windows/windows.py#L1623
*/
pub fn taylor(size: usize, η: f32, num_const_sidelobes: usize) -> Array1<f32> {
    taylor_from_coefficients(size, taylor_coefficients(η, num_const_sidelobes))
}

/**
This window is a tapered cosine window. It takes a fraction α between 0 and 1 and returns a window such that (1-α) is a rectangle window and the remainder is a cosine window. If α >= 1 this is equivalent to a hann window, if α <= 0 then this is equivalent to a rectangular window.
*/
pub fn tukey(size: usize, α: f32) -> Array1<f32> {
    let N = (size as f32) - 1.;

    let mut arr = Array1::zeros(size);
    for i in 0..size {
        let n = i as f32;
        if n >= 0. && n < α * N / 2. {
            arr[i] = 0.5 * (1. - (2. * PI * n / (α * N)).cos());
        } else if n >= α * N / 2. && n <= N / 2. {
            arr[i] = 1.;
        } else {
            arr[i] = arr[(size - 1) - i];
        }
    }

    arr
}

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use ndarray::{array, Array1};
    use num::traits::Pow;

    use crate::signal::fir::{
        hamming_window, taylor, taylor_coefficients, taylor_from_coefficients, tukey,
    };

    use super::dolph_chebychev;

    #[test]
    fn test_hamming() {
        // Test cases generated by python.
        // scipy.signals.windows.general_hamming(5, 25/46)
        let odd_test = array![0.08695652, 0.54347826, 1., 0.54347826, 0.08695652];
        let odd = hamming_window(5);

        assert_relative_eq!(odd, odd_test);

        // scipy.signals.windows.general_hamming(8, 25/46)
        let even_test = array![
            0.08695652, 0.25884161, 0.6450639, 0.95479014, 0.95479014, 0.6450639, 0.25884161,
            0.08695652
        ];
        let even = hamming_window(8);

        assert_relative_eq!(even, even_test);
    }

    #[test]
    fn test_dolph_cheby() {
        // Test cases generated by python.
        // scipy.signals.chebwin(5, 60)
        let odd_test = array![0.1876151f32, 0.68624124, 1., 0.68624124, 0.1876151];
        let odd = dolph_chebychev(5, 3.);

        assert_relative_eq!(odd, odd_test, epsilon = 1e-5);

        // scipy.signals.chebwin(6, 60)
        let even_test = array![0.13265103f32, 0.54770854, 1., 1., 0.54770854, 0.13265103];
        let even = dolph_chebychev(6, 3.);

        assert_relative_eq!(even, even_test);
    }

    #[test]
    fn test_taylor() {
        // Generated by extracted python code from scipy.signals.windows.taylor
        let coefficients_test = array![
            5.17689720e-01,
            4.12467758e-02,
            5.80987816e-04,
            -9.44310640e-04,
            6.16936336e-04,
            -3.35797035e-04,
            1.37942624e-04
        ];
        let sll = 10f32.pow(60. / 20.);
        let coefficients = taylor_coefficients(sll, 8);

        assert_relative_eq!(coefficients, coefficients_test,);

        // Test cases generated by python.
        // scipy.signals.windows.taylor(9, 8, 60)
        let odd_test = array![
            0.04254411, 0.20856778, 0.51995997, 0.85358958, 1., 0.85358958, 0.51995997, 0.20856778,
            0.04254411
        ];
        let odd = taylor(9, sll, 8);

        assert_relative_eq!(odd, odd_test,);

        let odd2 = taylor_from_coefficients(9, coefficients);

        assert_relative_eq!(odd2, odd);

        // scipy.signals.windows.taylor(14, 8, 60)
        let even_test = array![
            0.029335908198056322,
            0.10042175400497658,
            0.23575013925777444,
            0.432623287296759,
            0.6598005880848999,
            0.8632083936303664,
            0.9838888218868882,
            0.9838888218868882,
            0.8632083936303664,
            0.6598005880848999,
            0.432623287296759,
            0.23575013925777444,
            0.10042175400497658,
            0.029335908198056322
        ];
        let even = taylor(14, sll, 8);

        assert_relative_eq!(even, even_test, epsilon = 1e-5);
    }

    #[test]
    fn test_tukey() {
        // Test cases generated by python.
        // scipy.signals.windows.tukey(9, 0.8)
        let odd_test = array![
            0., 0.22221488, 0.69134172, 0.99039264, 1., 0.99039264, 0.69134172, 0.22221488, 0.
        ];
        let odd = tukey(9, 0.8);

        assert_relative_eq!(odd, odd_test,);

        // scipy.signals.windows.tukey(6, 0.8)
        let even_test = array![0., 0.5, 1., 1., 0.5, 0.];
        let even = tukey(6, 0.8);

        assert_relative_eq!(even, even_test);
    }
}
