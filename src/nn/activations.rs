use super::module::Module;
use crate::tensor::Tensor;

pub struct ReLu;
impl Module for ReLu {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    #[cfg(feature = "xla")]
    fn trace_xla(&self, builder: &xla::XlaBuilder, input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        crate::xla::XlaTraceable::trace(self, builder, input)
    }
}

pub struct Sigmoid;
impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.sigmoid()
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    #[cfg(feature = "xla")]
    fn trace_xla(&self, builder: &xla::XlaBuilder, input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        crate::xla::XlaTraceable::trace(self, builder, input)
    }
}

pub struct Tanh;
impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.tanh()
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    #[cfg(feature = "xla")]
    fn trace_xla(&self, builder: &xla::XlaBuilder, input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        crate::xla::XlaTraceable::trace(self, builder, input)
    }
}

pub struct LeakyReLu {
    pub alpha: f32,
}
impl Module for LeakyReLu {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.leaky_relu(self.alpha)
    }
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }

    #[cfg(feature = "xla")]
    fn trace_xla(&self, builder: &xla::XlaBuilder, input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        crate::xla::XlaTraceable::trace(self, builder, input)
    }
}

impl crate::xla::XlaTraceable for ReLu {
    #[cfg(feature = "xla")]
    fn trace(&self, _builder: &xla::XlaBuilder, input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        input.relu()
    }
}

impl crate::xla::XlaTraceable for Sigmoid {
    #[cfg(feature = "xla")]
    fn trace(&self, _builder: &xla::XlaBuilder, input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        crate::xla::XlaOps::sigmoid(input)
    }
}

impl crate::xla::XlaTraceable for Tanh {
    #[cfg(feature = "xla")]
    fn trace(&self, _builder: &xla::XlaBuilder, input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        input.tanh()
    }
}

impl crate::xla::XlaTraceable for LeakyReLu {
    #[cfg(feature = "xla")]
    fn trace(&self, _builder: &xla::XlaBuilder, input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        crate::xla::XlaOps::leaky_relu(input, self.alpha)
    }
}