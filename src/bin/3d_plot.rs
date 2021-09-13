/// This example illustrates how to add a custom attribute to a mesh and use it in a custom shader.
use radar_lib::{antenna::RectangularAperture, spherical_vis};

fn main() {
    let aperture = RectangularAperture {
        freq: 5e9,
        length_x: 0.1,
        length_y: 0.1,
    };
    spherical_vis::run(aperture);
}
