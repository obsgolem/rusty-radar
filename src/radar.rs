use std::f32::consts::PI;

use nalgebra::{UnitQuaternion, Vector3};
use ndarray::{Array1, Axis};
use num::traits::float::FloatCore;

use crate::{
    helper::{vec_to_aspect, wavelength},
    helper_traits::SphericalFunction,
    scene::PointTarget,
    signal::{realization::SignalRealization, scalar::Scalar, Pulse},
};

const c: f32 = 299_792_458.0;

pub const SPEED_OF_LIGHT: f32 = c;

fn doppler_to_radial_speed(f_0: f32, f_delta: f32) -> f32 {
    SPEED_OF_LIGHT / (2. * f_0)
}

pub struct Radar {
    pub pos: Vector3<f32>,
    pub vel: Vector3<f32>,
    pub acc: Vector3<f32>,
    // Represents the antenna frame to world frame transformation
    pub orient: UnitQuaternion<f32>,
    pub antenna_pattern: Box<dyn SphericalFunction>,
}

const BOLTZMANN_CONSTANT: f32 = 1.380649e-23; // J/K
const VACUUM_TEMP: f32 = 290.0; // K

pub fn range_equation(
    power_xmtd: f32,
    range: f32,
    xmtr_gain: f32,
    rcvr_gain: f32,
    signature: f32,
    freq: f32,
) -> f32 {
    let wavelength = SPEED_OF_LIGHT / freq;
    let effective_aperture = rcvr_gain * wavelength.powi(2) / (4. * PI);
    let density_at_tgt = power_xmtd * xmtr_gain / (4. * PI * range * range);
    let reflected_power = signature * density_at_tgt;
    let density_at_rcvr = reflected_power / (4. * PI * range * range);
    let rcvd_power = effective_aperture * density_at_rcvr;

    rcvd_power
}

// Pulsed radar data flow:
// Complex signal generation -> modulation (application of carrier frequency -> conversion to real signal) -> propogation ->
// demodulation (conversion to complex signal (includes low pass filter) -> removal of carrier frequency) ->
// sampling -> matched filter -> output
// Noise only gets a usable form after low pass filter, with variance proportional to the bandwidth of the low pass filter.

/*
The process of filtering of a recieved signal is performed by convolving the received signal with
another signal called the impulse response of the filter. The impulse response of the so called matched filter
of a transmitted signal is given by the time reversed and conjugated version of that signal.
Convolution itself involves a time reversal, so the two time reversals cancel out.
This is identical with the so called cross-correlation of the signal with itself.
For the matched filter we need to choose a point in the signal that we are trying to match. For our
purposes, we will use the t=0 point of the signal, not the end. This is only possible with the full dataset
available, as such a filter needs knowledge of the samples at times after the current input time (it is *non-causal*)
*/
pub fn apply_matched_filter<T: Scalar>(
    xmtd_signal: &SignalRealization<T>,
    rcvd_signal: &SignalRealization<T>,
) -> Array1<T> {
    let xmtd = xmtd_signal.signal_ref();
    let signal = rcvd_signal.signal();
    let mut filtered = signal.clone();
    for i in 0..signal.len() {
        let mut acc = T::zero();
        for j in 0..xmtd.len() {
            if i + j >= signal.len() {
                break;
            } else {
                acc += signal[i + j] * xmtd[j].conj();
            }
        }
        filtered[i] = acc;
    }

    filtered
}

// Given a transmitted signal and a geometry, performs the propogation of one pulse.
pub fn propogate_signal(mut signal: Pulse, radar: &Radar, target: &PointTarget) -> Pulse {
    let xmtr_to_tgt = target.pos - radar.pos;
    let xmtr_to_tgt_vel = target.vel - radar.vel;

    let range = xmtr_to_tgt.magnitude();
    let rel_vel = xmtr_to_tgt.normalize().dot(&xmtr_to_tgt_vel);

    let xmtr_to_tgt_antenna_frame = radar.orient.inverse_transform_vector(&xmtr_to_tgt);
    let tgt_to_xmtr_antenna_frame = radar
        .orient
        .inverse_transform_vector(&(radar.pos - target.pos));

    let gain = radar.antenna_pattern.lookup_vec(xmtr_to_tgt_antenna_frame);
    let signature = target.sig.lookup_vec(tgt_to_xmtr_antenna_frame);

    let rcvd_power = range_equation(
        signal.amplitude,
        range,
        gain,
        gain,
        signature,
        signal.clock.freq,
    );

    signal.amplitude *= rcvd_power.sqrt();
    signal.delay(2. * range / SPEED_OF_LIGHT);

    signal
}
