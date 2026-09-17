#[cfg(feature = "xla")]
use xla::{ElementType, XlaBuilder, XlaOp};

#[cfg(feature = "xla")]
pub trait XlaBuilderExt {
    fn parameter_f32(&self, index: i64, shape: &[i64], name: &str) -> Result<XlaOp, xla::Error>;
    fn constant_f32(&self, value: f32) -> Result<XlaOp, xla::Error>;
    fn constant_r1(&self, values: &[f32]) -> Result<XlaOp, xla::Error>;
}

#[cfg(feature = "xla")]
impl XlaBuilderExt for XlaBuilder {
    fn parameter_f32(&self, index: i64, shape: &[i64], name: &str) -> Result<XlaOp, xla::Error> {
        self.parameter(index, ElementType::F32, shape, name)
    }

    fn constant_f32(&self, value: f32) -> Result<XlaOp, xla::Error> {
        self.constant_r0(value)
    }

    fn constant_r1(&self, values: &[f32]) -> Result<XlaOp, xla::Error> {
        self.constant_r1(values)
    }
}

/// Helper functions for composing neural network layers in XLA computation graphs.
#[cfg(feature = "xla")]
pub struct XlaOps;

#[cfg(feature = "xla")]
impl XlaOps {
    /// Linear layer: output = input @ weights + bias (if present)
    pub fn linear(input: &XlaOp, weights: &XlaOp, bias: Option<&XlaOp>) -> Result<XlaOp, xla::Error> {
        let mm = input.dot(weights)?;
        if let Some(b) = bias {
            mm.add(b)
        } else {
            Ok(mm)
        }
    }

    /// ReLU activation: max(x, 0.0)
    pub fn relu(input: &XlaOp) -> Result<XlaOp, xla::Error> {
        input.relu()
    }

    /// Leaky ReLU: where(x > 0, x, x * alpha)
    pub fn leaky_relu(input: &XlaOp, alpha: f32) -> Result<XlaOp, xla::Error> {
        let builder = input.builder();
        let zero = builder.constant_r0(0.0f32)?;
        let alpha_op = builder.constant_r0(alpha)?;
        let cond = input.gt(&zero)?;
        let scaled = input.mul(&alpha_op)?;
        cond.select(input, &scaled)
    }

    /// Sigmoid activation: 1 / (1 + exp(-x))
    pub fn sigmoid(input: &XlaOp) -> Result<XlaOp, xla::Error> {
        let builder = input.builder();
        let one = builder.constant_r0(1.0f32)?;
        let neg = input.neg()?;
        let exp = neg.exp()?;
        let denom = one.add(&exp)?;
        one.div(&denom)
    }

    /// Tanh activation
    pub fn tanh(input: &XlaOp) -> Result<XlaOp, xla::Error> {
        input.tanh()
    }

    /// Mean Squared Error loss between prediction and target
    pub fn mse_loss(pred: &XlaOp, target: &XlaOp) -> Result<XlaOp, xla::Error> {
        let diff = pred.sub(target)?;
        let sq = diff.mul(&diff)?;
        sq.reduce_mean(&[], false)
    }
}
