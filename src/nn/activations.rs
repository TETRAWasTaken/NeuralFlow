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

    fn export_onnx(
        &self,
        ctx: &mut crate::utils::onnxexport::OnnxExportContext,
        input_name: &str,
    ) -> Result<String, crate::utils::onnxexport::OnnxExportError> {
        let out_name = ctx.next_name("relu_out");
        ctx.add_node(
            "Relu",
            vec![input_name.to_string()],
            vec![out_name.clone()],
            vec![],
        );
        Ok(out_name)
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

    fn export_onnx(
        &self,
        ctx: &mut crate::utils::onnxexport::OnnxExportContext,
        input_name: &str,
    ) -> Result<String, crate::utils::onnxexport::OnnxExportError> {
        let out_name = ctx.next_name("sigmoid_out");
        ctx.add_node(
            "Sigmoid",
            vec![input_name.to_string()],
            vec![out_name.clone()],
            vec![],
        );
        Ok(out_name)
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

    fn export_onnx(
        &self,
        ctx: &mut crate::utils::onnxexport::OnnxExportContext,
        input_name: &str,
    ) -> Result<String, crate::utils::onnxexport::OnnxExportError> {
        let out_name = ctx.next_name("tanh_out");
        ctx.add_node(
            "Tanh",
            vec![input_name.to_string()],
            vec![out_name.clone()],
            vec![],
        );
        Ok(out_name)
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

    fn export_onnx(
        &self,
        ctx: &mut crate::utils::onnxexport::OnnxExportContext,
        input_name: &str,
    ) -> Result<String, crate::utils::onnxexport::OnnxExportError> {
        let out_name = ctx.next_name("leaky_relu_out");
        ctx.add_node(
            "LeakyRelu",
            vec![input_name.to_string()],
            vec![out_name.clone()],
            vec![crate::utils::onnxexport::InternalAttribute::Float(
                "alpha".to_string(),
                self.alpha,
            )],
        );
        Ok(out_name)
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