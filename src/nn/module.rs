use crate::tensor::Tensor;
use std::cell::Cell;

pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor>;

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    fn set_training(&self, _mode: bool) {}

    fn train(&self) {
        self.set_training(true);
    }

    fn eval(&self) {
        self.set_training(false);
    }

    fn to(&self, device: crate::tensor::Device) {
        for p in self.parameters() {
            p.to(device);
        }
    }

    #[cfg(feature = "xla")]
    fn trace_xla(&self, _builder: &xla::XlaBuilder, _input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        Err(xla::Error::General("This module has not implemented trace_xla".into()))
    }

    fn export_onnx(
        &self,
        _ctx: &mut crate::utils::onnxexport::OnnxExportContext,
        _input_name: &str,
    ) -> Result<String, crate::utils::onnxexport::OnnxExportError> {
        Err(crate::utils::onnxexport::OnnxExportError::UnsupportedLayer(
            std::any::type_name::<Self>().to_string(),
        ))
    }
}

pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
    training: Cell<bool>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self {
            layers,
            training: Cell::new(true),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            layers: Vec::with_capacity(capacity),
            training: Cell::new(true),
        }
    }

    pub fn add(&mut self, module: impl Module + 'static) {
        self.layers.push(Box::new(module));
    }

    pub fn layers(&self) -> &[Box<dyn Module>] {
        &self.layers
    }

    pub fn is_training(&self) -> bool {
        self.training.get()
    }

    pub fn save_onnx(
        &self,
        input_shape: &[usize],
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), crate::utils::onnxexport::OnnxExportError> {
        crate::utils::onnxexport::export_to_onnx(self, input_shape, path)
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.layers.iter().fold(input.clone(), |acc, layer| layer.forward(&acc))
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn set_training(&self, mode: bool) {
        self.training.set(mode);
        for layer in &self.layers {
            layer.set_training(mode);
        }
    }

    #[cfg(feature = "xla")]
    fn trace_xla(&self, builder: &xla::XlaBuilder, input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        let mut curr = input.clone();
        for layer in &self.layers {
            curr = layer.trace_xla(builder, &curr)?;
        }
        Ok(curr)
    }

    fn export_onnx(
        &self,
        ctx: &mut crate::utils::onnxexport::OnnxExportContext,
        input_name: &str,
    ) -> Result<String, crate::utils::onnxexport::OnnxExportError> {
        let mut curr = input_name.to_string();
        for layer in &self.layers {
            curr = layer.export_onnx(ctx, &curr)?;
        }
        Ok(curr)
    }
}

impl crate::xla::XlaTraceable for Sequential {
    #[cfg(feature = "xla")]
    fn trace(&self, builder: &xla::XlaBuilder, input: &xla::XlaOp) -> Result<xla::XlaOp, xla::Error> {
        self.trace_xla(builder, input)
    }
}