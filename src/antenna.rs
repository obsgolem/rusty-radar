use std::f32::consts::PI;

use crate::{helper_traits::SphericalFunction, radar::SPEED_OF_LIGHT, signal::sinc};

// Represents an antenna pattern of uniformly illuminated rectangular aperture.
// Such a pattern is given by examining the far field diffraction pattern of the aperture
// assuming it is illuminated by a plane wave of the given frequency.
// Sources:
// [1] https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-013-electromagnetics-and-applications-spring-2009/readings/MIT6_013S09_chap11.pdf
pub struct RectangularAperture {
    pub freq: f32,
    pub length_x: f32,
    pub length_y: f32,
}

const c: f32 = SPEED_OF_LIGHT;

impl SphericalFunction for RectangularAperture {
    fn lookup(&self, az: f32, el: f32) -> f32 {
        let area = self.length_x * self.length_y;

        // Wavelength
        let λ = c / self.freq;
        // Equation 11.1.17 in [1]
        area * 4. * PI / λ
            * sinc(PI * az * self.length_x / λ).powi(2)
            * sinc(PI * el * self.length_y / λ).powi(2)
    }
}
