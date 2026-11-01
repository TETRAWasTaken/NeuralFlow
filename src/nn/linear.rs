use super::module::Module;
use crate::tensor::Tensor;

pub struct Linear {
    pub weights: Tensor,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weights: Tensor::kaiming_normal((in_features, out_features)),
        }
    }

    pub fn random(in_features: usize, out_features: usize) -> Self {
        Self {
            weights: Tensor::random((in_features, out_features)),
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        input.matmul(&self.weights)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone()]
    }

    #[cfg(feature = "xla")]
    fn trace_xla(&self, builder: &xla::XlaBuilder, input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        crate::xla::XlaTraceable::trace(self, builder, input)
    }

    fn export_onnx(
        &self,
        ctx: &mut crate::utils::onnxexport::OnnxExportContext,
        input_name: &str,
    ) -> Result<String, crate::utils::onnxexport::OnnxExportError> {
        let w_name = ctx.next_name("linear_w");
        let w_inner = self.weights.0.borrow();
        ctx.add_initializer(
            w_name.clone(),
            vec![w_inner.shape.0 as i64, w_inner.shape.1 as i64],
            w_inner.data.clone(),
        );

        let out_name = ctx.next_name("linear_out");
        ctx.add_node(
            "MatMul",
            vec![input_name.to_string(), w_name],
            vec![out_name.clone()],
            vec![],
        );

        Ok(out_name)
    }
}

impl crate::xla::XlaTraceable for Linear {
    #[cfg(feature = "xla")]
    fn trace(&self, builder: &xla::XlaBuilder, input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        let w = self.weights.0.borrow();
        let w_op = builder.constant_r2(&w.data, w.shape.0 as i64, w.shape.1 as i64)?;
        input.dot(&w_op)
    }
}