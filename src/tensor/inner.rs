use std::cell::{Cell, RefCell};
use std::rc::Rc;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

pub fn set_grad_enabled(enabled: bool) -> bool {
    GRAD_ENABLED.with(|g| g.replace(enabled))
}

pub struct NoGradGuard {
    prev: bool,
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev);
    }
}

pub fn no_grad() -> NoGradGuard {
    let prev = set_grad_enabled(false);
    NoGradGuard { prev }
}

#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<TensorInner>>);

pub struct TensorInner {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub shape: (usize, usize),
    pub backward: Option<Box<dyn Fn()>>,
    pub prev: Vec<Tensor>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: (usize, usize)) -> Self {
        let n = data.len();
        assert!(n == shape.0 * shape.1, "Data and shape must match");

        Tensor(Rc::new(RefCell::new(TensorInner {
            data,
            grad: vec![0.0; n],
            shape,
            backward: None,
            prev: vec![],
        })))
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        Self::new(vec![0.0; shape.0 * shape.1], shape)
    }

    pub fn random(shape: (usize, usize)) -> Self {
        let len = shape.0 * shape.1;
        let data = (0..len).map(|_| (rand::random::<f32>() * 2.0) - 1.0).collect();
        Self::new(data, shape)
    }

    pub fn zero_grad(&self) {
        let mut inner = self.0.borrow_mut();
        for g in inner.grad.iter_mut() {
            *g = 0.0;
        }
    }
}