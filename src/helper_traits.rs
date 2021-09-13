use nalgebra::Vector3;

use crate::helper::vec_to_aspect;

// A function defined on the surface of a sphere `S`
pub trait SphericalFunction {
    fn lookup(&self, az: f32, el: f32) -> f32;
    fn lookup_vec(&self, vec: Vector3<f32>) -> f32 {
        let (az, el) = vec_to_aspect(vec);
        self.lookup(az, el)
    }
}

pub trait SphericalFunctionHelper {
    fn lookup_many(&self, items: impl Iterator<Item = Vector3<f32>>) -> Vec<f32>;
}

impl<T: SphericalFunction> SphericalFunctionHelper for T {
    fn lookup_many(&self, items: impl Iterator<Item = Vector3<f32>>) -> Vec<f32> {
        items.map(|x| self.lookup_vec(x)).collect()
    }
}

// A function defined on `SxS`, the cartesian product of `S` with itself
pub trait S2Function {
    fn lookup(&self, az1: f32, el1: f32, az2: f32, el2: f32) -> f32;
}
