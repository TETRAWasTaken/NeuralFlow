use std::fmt;
use std::ops::Deref;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub(crate) Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Shape(dims.into())
    }

    pub fn d1(d0: usize) -> Self {
        Shape(vec![d0])
    }

    pub fn d2(d0: usize, d1: usize) -> Self {
        Shape(vec![d0, d1])
    }

    pub fn d3(d0: usize, d1: usize, d2: usize) -> Self {
        Shape(vec![d0, d1, d2])
    }

    pub fn d4(d0: usize, d1: usize, d2: usize, d3: usize) -> Self {
        Shape(vec![d0, d1, d2, d3])
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn numel(&self) -> usize {
        if self.0.is_empty() {
            0
        } else {
            self.0.iter().product()
        }
    }

    pub fn as_2d(&self) -> Option<(usize, usize)> {
        match self.0.as_slice() {
            [r, c] => Some((*r, *c)),
            _ => None,
        }
    }

    pub fn as_4d(&self) -> Option<(usize, usize, usize, usize)> {
        match self.0.as_slice() {
            [n, c, h, w] => Some((*n, *c, *h, *w)),
            _ => None,
        }
    }
}

impl Deref for Shape {
    type Target = [usize];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<(usize, usize)> for Shape {
    fn from(tuple: (usize, usize)) -> Self {
        Shape::d2(tuple.0, tuple.1)
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from(tuple: (usize, usize, usize, usize)) -> Self {
        Shape::d4(tuple.0, tuple.1, tuple.2, tuple.3)
    }
}

impl From<&[usize]> for Shape {
    fn from(slice: &[usize]) -> Self {
        Shape(slice.to_vec())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(vec: Vec<usize>) -> Self {
        Shape(vec)
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.0)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.0.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
    }
}
