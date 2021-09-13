use nalgebra::{UnitQuaternion, Vector3};
use ndarray::Array2;

use crate::helper_traits::SphericalFunction;

pub struct PointTarget {
    pub pos: Vector3<f32>,
    pub vel: Vector3<f32>,
    pub acc: Vector3<f32>,
    pub sig: RadarSignature,
}
pub enum Object {
    PointTarget(PointTarget),
}

pub struct Scene(Vec<Object>);

pub enum RadarSignature {
    Constant(f32),
}

impl SphericalFunction for RadarSignature {
    fn lookup(&self, _: f32, _: f32) -> f32 {
        match *self {
            RadarSignature::Constant(x) => x,
        }
    }
}

// Integrate the equations of motion to get an estimate of the target position and velocity after a timestep dt.
pub fn dead_reckon_accelerating(
    mut pos: Vector3<f32>,
    mut vel: Vector3<f32>,
    acc: Vector3<f32>,
    dt: f32,
) -> (Vector3<f32>, Vector3<f32>) {
    // Semi-implicit euler, see e.g. https://gafferongames.com/post/integration_basics/
    vel += dt * acc;
    pos += dt * vel;
    (pos, vel)
}

// Integrate the equations of motion to get an estimate of the target position after a timestep dt.
pub fn dead_reckon(mut pos: Vector3<f32>, vel: Vector3<f32>, dt: f32) -> Vector3<f32> {
    pos += dt * vel;
    pos
}
