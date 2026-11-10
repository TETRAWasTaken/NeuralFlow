use std::fmt;
use std::fs;
use std::path::Path;
use onnx_rs::ast::*;

#[derive(Debug)]
pub enum OnnxExportError {
    UnsupportedLayer(String),
    Io(std::io::Error),
    InvalidShape(String),
}

impl fmt::Display for OnnxExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxExportError::UnsupportedLayer(msg) => {
                write!(f, "Unsupported layer for ONNX export: {}", msg)
            }
            OnnxExportError::Io(err) => write!(f, "I/O error: {}", err),
            OnnxExportError::InvalidShape(msg) => write!(f, "Invalid shape: {}", msg),
        }
    }
}

impl std::error::Error for OnnxExportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OnnxExportError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for OnnxExportError {
    fn from(err: std::io::Error) -> Self {
        OnnxExportError::Io(err)
    }
}

pub enum InternalAttribute {
    Int(String, i64),
    Float(String, f32),
    Ints(String, Vec<i64>),
    Floats(String, Vec<f32>),
    String(String, String),
}

pub struct InternalNode {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<InternalAttribute>,
}

pub struct InternalTensor {
    pub name: String,
    pub dims: Vec<i64>,
    pub data: Vec<f32>,
}

#[derive(Default)]
pub struct OnnxExportContext {
    pub nodes: Vec<InternalNode>,
    pub initializers: Vec<InternalTensor>,
    name_counter: usize,
}

impl OnnxExportContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next_name(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.name_counter);
        self.name_counter += 1;
        name
    }

    pub fn add_initializer(&mut self, name: String, dims: Vec<i64>, data: Vec<f32>) {
        self.initializers.push(InternalTensor { name, dims, data });
    }

    pub fn add_node(
        &mut self,
        op_type: &str,
        inputs: Vec<String>,
        outputs: Vec<String>,
        attributes: Vec<InternalAttribute>,
    ) -> String {
        let name = format!("{}_{}", op_type.to_lowercase(), self.name_counter);
        self.name_counter += 1;
        self.nodes.push(InternalNode {
            name: name.clone(),
            op_type: op_type.to_string(),
            inputs,
            outputs,
            attributes,
        });
        name
    }
}

/// Exports a Rensor module/sequential graph into ONNX format.
pub fn export_to_onnx(
    model: &crate::nn::Sequential,
    input_shape: &[usize],
    path: impl AsRef<Path>,
) -> Result<(), OnnxExportError> {
    if input_shape.is_empty() {
        return Err(OnnxExportError::InvalidShape("Input shape cannot be empty".into()));
    }

    let mut ctx = OnnxExportContext::new();
    let graph_input_name = "input_0".to_string();

    // Trace the sequential layers via the Module trait method
    use crate::nn::Module;
    let graph_output_name = model.export_onnx(&mut ctx, &graph_input_name)?;

    // Build ONNX input ValueInfo
    let input_dim_protos: Vec<TensorShapeDimension> = input_shape
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if i == 0 {
                // Dynamic batch size
                TensorShapeDimension {
                    value: Dimension::Param("batch"),
                    ..Default::default()
                }
            } else {
                TensorShapeDimension {
                    value: Dimension::Value(d as i64),
                    ..Default::default()
                }
            }
        })
        .collect();

    let input_vi = ValueInfo {
        name: &graph_input_name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: Some(TensorShape {
                    dim: input_dim_protos,
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    };

    // Build ONNX output ValueInfo with dynamic batch and inferred output feature dimension
    let output_vi = ValueInfo {
        name: &graph_output_name,
        r#type: Some(TypeProto {
            value: Some(TypeValue::Tensor(TensorTypeProto {
                elem_type: DataType::Float,
                shape: Some(TensorShape {
                    dim: vec![
                        TensorShapeDimension {
                            value: Dimension::Param("batch"),
                            ..Default::default()
                        },
                        TensorShapeDimension::default(),
                    ],
                }),
            })),
            ..Default::default()
        }),
        ..Default::default()
    };

    // Convert initializers
    let initializers: Vec<TensorProto> = ctx
        .initializers
        .iter()
        .map(|init| {
            TensorProto::from_f32(&init.name, init.dims.clone(), init.data.clone())
        })
        .collect();

    // Convert nodes
    let nodes: Vec<Node> = ctx
        .nodes
        .iter()
        .map(|n| {
            let attributes: Vec<Attribute> = n
                .attributes
                .iter()
                .map(|attr| match attr {
                    InternalAttribute::Int(name, val) => Attribute {
                        name,
                        r#type: AttributeType::Int,
                        i: *val,
                        ..Default::default()
                    },
                    InternalAttribute::Float(name, val) => Attribute {
                        name,
                        r#type: AttributeType::Float,
                        f: *val,
                        ..Default::default()
                    },
                    InternalAttribute::Ints(name, vals) => Attribute {
                        name,
                        r#type: AttributeType::Ints,
                        ints: vals.clone(),
                        ..Default::default()
                    },
                    InternalAttribute::Floats(name, vals) => Attribute {
                        name,
                        r#type: AttributeType::Floats,
                        floats: vals.clone(),
                        ..Default::default()
                    },
                    InternalAttribute::String(name, val) => Attribute {
                        name,
                        r#type: AttributeType::String,
                        s: val.as_bytes(),
                        ..Default::default()
                    },
                })
                .collect();

            let op_type = OpType::from(n.op_type.as_str());

            Node {
                name: &n.name,
                op_type,
                input: n.inputs.iter().map(|s| s.as_str()).collect(),
                output: n.outputs.iter().map(|s| s.as_str()).collect(),
                attribute: attributes,
                ..Default::default()
            }
        })
        .collect();

    let graph = Graph {
        name: "rensor_graph",
        node: nodes,
        initializer: initializers,
        input: vec![input_vi],
        output: vec![output_vi],
        ..Default::default()
    };

    let model = Model {
        ir_version: 9,
        opset_import: vec![OperatorSetId {
            domain: "",
            version: 19,
        }],
        producer_name: "Rensor",
        producer_version: "1.1.0",
        graph: Some(graph),
        ..Default::default()
    };

    let encoded = onnx_rs::encode(&model);
    fs::write(path, encoded)?;

    Ok(())
}
