use ndarray::{Array1, Axis};

use crate::signal::scalar::Scalar;

pub trait Pad {
    type Data: Scalar;
    fn pad_back(self, with: Self::Data, num: usize) -> Array1<Self::Data>;
    fn pad_front(self, with: Self::Data, num: usize) -> Array1<Self::Data>;
}

impl<T: Scalar> Pad for Array1<T> {
    type Data = T;

    fn pad_back(mut self, with: Self::Data, num: usize) -> Array1<Self::Data> {
        self.append(Axis(0), Array1::from_elem((num,), with).view())
            .unwrap();
        self
    }

    fn pad_front(self, with: Self::Data, num: usize) -> Array1<Self::Data> {
        let mut arr = Array1::from_elem((num,), with);
        arr.append(Axis(0), self.view()).unwrap();
        arr
    }
}
