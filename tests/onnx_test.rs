use onnx_rs::ast::OpType;
use rensor::prelude::*;
use std::fs;

#[test]
fn test_export_mlp_to_onnx_and_parse_back() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(784, 128)),
        Box::new(ReLu),
        Box::new(Linear::new(128, 10)),
        Box::new(Sigmoid),
    ]);

    let temp_dir = std::env::temp_dir();
    let onnx_path = temp_dir.join("test_mlp_model.onnx");

    // Export model
    model
        .save_onnx(&[1, 784], &onnx_path)
        .expect("Failed to export MLP to ONNX");

    assert!(onnx_path.exists(), "ONNX file was not created");

    // Read back and parse using onnx-rs
    let bytes = fs::read(&onnx_path).expect("Failed to read exported ONNX file");
    assert!(!bytes.is_empty(), "Exported ONNX file is empty");

    let parsed_model = onnx_rs::parse(&bytes).expect("Failed to parse ONNX protobuf with onnx-rs");

    assert_eq!(parsed_model.producer_name, "Rensor");
    assert_eq!(parsed_model.ir_version, 9);

    let graph = parsed_model.graph.as_ref().expect("Model graph missing");
    assert_eq!(graph.name, "rensor_graph");

    // Verify graph inputs and outputs
    assert_eq!(graph.input.len(), 1);
    assert_eq!(graph.input[0].name, "input_0");
    assert_eq!(graph.output.len(), 1);

    // Verify initializers (2 weight matrices for 2 Linear layers)
    assert_eq!(graph.initializer.len(), 2);
    assert_eq!(graph.initializer[0].dims(), &[784, 128]);
    assert_eq!(graph.initializer[1].dims(), &[128, 10]);

    // Verify nodes: MatMul -> Relu -> MatMul -> Sigmoid
    assert_eq!(graph.node.len(), 4);
    assert_eq!(graph.node[0].op_type, OpType::MatMul);
    assert_eq!(graph.node[1].op_type, OpType::Relu);
    assert_eq!(graph.node[2].op_type, OpType::MatMul);
    assert_eq!(graph.node[3].op_type, OpType::Sigmoid);

    // Clean up
    let _ = fs::remove_file(onnx_path);
}

#[test]
fn test_export_cnn_to_onnx_and_parse_back() {
    let model = Sequential::new(vec![
        Box::new(Conv2d::new_with_options(
            1,
            8,
            (3, 3),
            (1, 1),
            (1, 1),
            true, // use bias
        )),
        Box::new(ReLu),
        Box::new(MaxPool2d::new((2, 2))),
        Box::new(Flatten::new()),
        Box::new(Linear::new(8 * 14 * 14, 10)),
    ]);

    let temp_dir = std::env::temp_dir();
    let onnx_path = temp_dir.join("test_cnn_model.onnx");

    // Export CNN with input shape [batch, 1, 28, 28]
    export_to_onnx(&model, &[1, 1, 28, 28], &onnx_path)
        .expect("Failed to export CNN to ONNX");

    assert!(onnx_path.exists(), "ONNX file was not created");

    // Read back and parse
    let bytes = fs::read(&onnx_path).expect("Failed to read exported CNN file");
    let parsed_model = onnx_rs::parse(&bytes).expect("Failed to parse ONNX protobuf with onnx-rs");

    let graph = parsed_model.graph.as_ref().expect("Graph missing");

    // Initializers: Conv weight, Conv bias, Linear weight
    assert_eq!(graph.initializer.len(), 3);
    assert_eq!(graph.initializer[0].dims(), &[8, 1, 3, 3]); // Conv weight
    assert_eq!(graph.initializer[1].dims(), &[8]);          // Conv bias
    assert_eq!(graph.initializer[2].dims(), &[8 * 14 * 14, 10]); // Linear weight

    // Nodes: Conv -> Relu -> MaxPool -> Flatten -> MatMul
    assert_eq!(graph.node.len(), 5);
    assert_eq!(graph.node[0].op_type, OpType::Conv);
    assert_eq!(graph.node[1].op_type, OpType::Relu);
    assert_eq!(graph.node[2].op_type, OpType::MaxPool);
    assert_eq!(graph.node[3].op_type, OpType::Flatten);
    assert_eq!(graph.node[4].op_type, OpType::MatMul);

    // Verify Conv node attributes (kernel_shape, strides, pads)
    let conv_node = &graph.node[0];
    let kernel_attr = conv_node.attribute.iter().find(|a| a.name == "kernel_shape");
    assert!(kernel_attr.is_some());
    assert_eq!(kernel_attr.unwrap().ints, &[3, 3]);

    let pads_attr = conv_node.attribute.iter().find(|a| a.name == "pads");
    assert!(pads_attr.is_some());
    assert_eq!(pads_attr.unwrap().ints, &[1, 1, 1, 1]);

    // Clean up
    let _ = fs::remove_file(onnx_path);
}
