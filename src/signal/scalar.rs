use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Mul, MulAssign},
};

use num::{complex::Complex32, Float};

pub trait Scalar:
    Copy + Mul<Self, Output = Self> + Add<Self, Output = Self> + AddAssign + MulAssign + Debug + Display
{
    type CompType: Float + Scalar;
    fn norm_sqr(&self) -> Self::CompType;
    fn norm(&self) -> Self::CompType;
    fn conj(&self) -> Self;
    fn zero() -> Self;
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

    fn zero() -> Self {
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

    fn zero() -> Self {
        0.0.into()
    }
}
