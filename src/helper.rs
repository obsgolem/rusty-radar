use nalgebra::Vector3;

use crate::radar::SPEED_OF_LIGHT;

// Given a vector in a local frame, compute the azimuth and elevation.
// x is taken as forward, y as right, z as down. Positive azimuth goes from x to y, positive elevation goes from x to -z.
pub fn vec_to_aspect(vec: Vector3<f32>) -> (f32, f32) {
    let az = f32::atan2(vec[1], vec[0]);
    let el = f32::asin(vec[2] / vec.magnitude());
    (az, el)
}

// Computes a vector in a local frame given azimuth and elevation.
// x is taken as forward, y as right, z as down. Positive azimuth goes from x to y, positive elevation goes from x to -z.
pub fn aspect_to_vec(az: f32, el: f32) -> Vector3<f32> {
    Vector3::new(az.cos() * el.cos(), az.sin() * el.cos(), el.sin())
}

pub fn wavelength(f: f32) -> f32 {
    SPEED_OF_LIGHT / f
}

pub fn decibels(x: f32) -> f32 {
    10. * x.log10()
}

pub fn decibels_or_else(x: f32, or: f32) -> f32 {
    if x <= 0. {
        or
    } else {
        10. * x.log10()
    }
}

pub fn normalize_all(mut slice: impl AsMut<[Vector3<f32>]>) {
    for x in slice.as_mut().iter_mut() {
        x.normalize_mut();
    }
}
