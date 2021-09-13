use nalgebra::{UnitQuaternion, Vector3};
use num::Float;
use radar_lib::{
    antenna::RectangularAperture,
    radar::{propogate_signal, Radar},
    scene::PointTarget,
    scene::RadarSignature,
    signal::{Pulse, PulseType},
};

fn main() {
    let tgt = PointTarget {
        pos: Vector3::new(1., 0., 1.),
        vel: Vector3::new(0., 0., 0.),
        acc: Vector3::new(0., 0., 0.),
        sig: RadarSignature::Constant(10.),
    };

    let radar = Radar {
        pos: Vector3::new(0., 0., 0.),
        vel: Vector3::new(0., 0., 0.),
        acc: Vector3::new(0., 0., 0.),
        orient: UnitQuaternion::from_euler_angles(0., -45.0.to_radians(), 0.0),
        antenna_pattern: Box::new(RectangularAperture {
            freq: 1.,
            length_x: 1.,
            length_y: 1.,
        }),
    };

    let result = propogate_signal(
        Pulse {
            clock: 1.0.into(),
            length: 0.1,
            start_time: 0.,
            amplitude: 1.,
            sig_type: PulseType::Rectangle,
        },
        &radar,
        &tgt,
    );
}
