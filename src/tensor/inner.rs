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

use super::device::Device;
use super::shape::Shape;

#[derive(Clone)]
pub struct Tensor(pub Rc<RefCell<TensorInner>>);

pub struct TensorInner {
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
    pub shape: (usize, usize),
    pub shape4d: Option<(usize, usize, usize, usize)>,
    pub backward: Option<Box<dyn Fn()>>,
    pub prev: Vec<Tensor>,
    pub device: Device,
    #[cfg(feature = "xla")]
    pub xla_buffer: Option<xla::PjRtBuffer>,
    #[cfg(feature = "xla")]
    pub xla_grad: Option<xla::PjRtBuffer>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: (usize, usize)) -> Self {
        let n = data.len();
        assert!(n == shape.0 * shape.1, "Data and shape must match");

        Tensor(Rc::new(RefCell::new(TensorInner {
            data,
            grad: vec![0.0; n],
            shape,
            shape4d: None,
            backward: None,
            prev: vec![],
            device: Device::Cpu,
            #[cfg(feature = "xla")]
            xla_buffer: None,
            #[cfg(feature = "xla")]
            xla_grad: None,
        })))
    }

    pub fn new_4d(data: Vec<f32>, shape: (usize, usize, usize, usize)) -> Self {
        let n = data.len();
        let expected = shape.0 * shape.1 * shape.2 * shape.3;
        assert_eq!(n, expected, "Data length {} does not match 4D shape {:?}", n, shape);

        Tensor(Rc::new(RefCell::new(TensorInner {
            data,
            grad: vec![0.0; n],
            shape: (shape.0, shape.1 * shape.2 * shape.3),
            shape4d: Some(shape),
            backward: None,
            prev: vec![],
            device: Device::Cpu,
            #[cfg(feature = "xla")]
            xla_buffer: None,
            #[cfg(feature = "xla")]
            xla_grad: None,
        })))
    }

    pub fn shape4d(&self) -> Option<(usize, usize, usize, usize)> {
        self.0.borrow().shape4d
    }

    pub fn shape(&self) -> (usize, usize) {
        self.0.borrow().shape
    }

    pub fn reshape_4d(&self, shape: (usize, usize, usize, usize)) -> Self {
        let n = self.0.borrow().data.len();
        let expected = shape.0 * shape.1 * shape.2 * shape.3;
        assert_eq!(
            n, expected,
            "Cannot reshape tensor with {} elements to 4D shape {:?}",
            n, shape
        );

        let out = Self::new_4d(self.0.borrow().data.clone(), shape);

        if is_grad_enabled() {
            out.0.borrow_mut().prev = vec![self.clone()];
            let self_clone = self.clone();
            let out_clone = out.clone();

            out.0.borrow_mut().backward = Some(Box::new(move || {
                let out_inner = out_clone.0.borrow();
                let out_grad = &out_inner.grad;
                let mut self_inner = self_clone.0.borrow_mut();
                for (sg, &og) in self_inner.grad.iter_mut().zip(out_grad.iter()) {
                    *sg += og;
                }
            }));
        }

        out
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        Self::new(vec![0.0; shape.0 * shape.1], shape)
    }

    pub fn zeros_4d(shape: (usize, usize, usize, usize)) -> Self {
        Self::new_4d(vec![0.0; shape.0 * shape.1 * shape.2 * shape.3], shape)
    }

    pub fn random_4d(shape: (usize, usize, usize, usize)) -> Self {
        let len = shape.0 * shape.1 * shape.2 * shape.3;
        let data = (0..len).map(|_| (rand::random::<f32>() * 2.0) - 1.0).collect();
        Self::new_4d(data, shape)
    }

    pub fn random(shape: (usize, usize)) -> Self {
        let len = shape.0 * shape.1;
        let data = (0..len).map(|_| (rand::random::<f32>() * 2.0) - 1.0).collect();
        Self::new(data, shape)
    }

    pub fn kaiming_normal(shape: (usize, usize)) -> Self {
        let (fan_in, _) = shape;
        assert!(fan_in > 0, "fan_in must be greater than 0");
        let std_dev = (2.0 / fan_in as f32).sqrt();
        let len = shape.0 * shape.1;
        let mut data = Vec::with_capacity(len);

        while data.len() < len {
            // Box-Muller transform for normal distribution
            let u1 = rand::random::<f32>().max(f32::EPSILON);
            let u2 = rand::random::<f32>();

            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f32::consts::PI * u2;

            let z0 = r * theta.cos() * std_dev;
            let z1 = r * theta.sin() * std_dev;

            data.push(z0);
            if data.len() < len {
                data.push(z1);
            }
        }

        Self::new(data, shape)
    }

    pub fn kaiming_uniform(shape: (usize, usize)) -> Self {
        let (fan_in, _) = shape;
        assert!(fan_in > 0, "fan_in must be greater than 0");
        let bound = (6.0 / fan_in as f32).sqrt();
        let len = shape.0 * shape.1;

        let data = (0..len)
            .map(|_| (rand::random::<f32>() * 2.0 - 1.0) * bound)
            .collect();

        Self::new(data, shape)
    }

    pub fn zero_grad(&self) {
        let mut inner = self.0.borrow_mut();
        for g in inner.grad.iter_mut() {
            *g = 0.0;
        }
    }

    pub fn device(&self) -> Device {
        self.0.borrow().device
    }

    pub fn unified_shape(&self) -> Shape {
        let inner = self.0.borrow();
        if let Some(s4) = inner.shape4d {
            Shape::d4(s4.0, s4.1, s4.2, s4.3)
        } else {
            Shape::d2(inner.shape.0, inner.shape.1)
        }
    }

    pub fn numel(&self) -> usize {
        self.0.borrow().data.len()
    }

    pub fn to(&self, device: Device) -> Self {
        if self.device() == device {
            return self.clone();
        }
        match device {
            Device::Cpu => self.to_cpu(),
            Device::Gpu(_) | Device::Tpu(_) => {
                #[cfg(feature = "xla")]
                {
                    let mut inner = self.0.borrow_mut();
                    inner.device = device;
                    drop(inner);
                    self.clone()
                }
                #[cfg(not(feature = "xla"))]
                {
                    panic!(
                        "Targeting device {:?} requires the `xla` feature to be enabled in NeuralFlow.",
                        device
                    );
                }
            }
        }
    }

    pub fn to_cpu(&self) -> Self {
        let inner = self.0.borrow();
        if inner.device == Device::Cpu {
            return self.clone();
        }

        #[cfg(feature = "xla")]
        {
            if let Some(ref buf) = inner.xla_buffer {
                let literal = buf.to_literal_sync().expect("Failed to sync XLA buffer to CPU");
                let data: Vec<f32> = literal.to_vec().expect("Failed to copy XLA literal to Vec<f32>");
                drop(inner);
                let out = if let Some(s4) = self.shape4d() {
                    Self::new_4d(data, s4)
                } else {
                    Self::new(data, self.shape())
                };
                return out;
            }
        }

        let mut inner = self.0.borrow_mut();
        inner.device = Device::Cpu;
        drop(inner);
        self.clone()
    }

    pub fn to_device(&self, device: Device) -> Self {
        self.to(device)
    }
}