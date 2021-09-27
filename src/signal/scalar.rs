use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Mul, MulAssign},
};

use num::{complex::Complex32, Float, Num};

pub trait Scalar:
    Copy
    + Mul<Self, Output = Self>
    + Add<Self, Output = Self>
    + AddAssign
    + MulAssign
    + Debug
    + Display
    + Num
{
    type CompType: Float + Scalar;
    fn norm_sqr(&self) -> Self::CompType;
    fn norm(&self) -> Self::CompType;
    fn conj(&self) -> Self;
    fn re(&self) -> Self::CompType;
    fn im(&self) -> Self::CompType;
}

impl Scalar for f32 {
    type CompType = f32;

    fn norm_sqr(&self) -> Self::CompType {
        self.abs().powi(2)
    }

    fn norm(&self) -> Self::CompType {
        self.abs()
    }

    fn conj(&self) -> Self {
        *self
    }

    fn re(&self) -> Self::CompType {
        *self
    }

    fn im(&self) -> Self::CompType {
        0.
    }
}

impl Scalar for Complex32 {
    type CompType = f32;

    fn norm_sqr(&self) -> Self::CompType {
        self.norm_sqr()
    }

    fn norm(&self) -> Self::CompType {
        Complex32::norm(*self)
    }

    fn conj(&self) -> Self {
        Complex32::conj(self)
    }

    fn re(&self) -> Self::CompType {
        self.re
    }

    fn im(&self) -> Self::CompType {
        self.im
    }
}
