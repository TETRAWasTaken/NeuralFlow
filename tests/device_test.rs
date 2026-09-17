use NeuralFlow::prelude::*;

#[test]
fn test_device_properties_and_display() {
    let cpu = Device::Cpu;
    assert!(cpu.is_cpu());
    assert!(!cpu.is_gpu());
    assert!(!cpu.is_tpu());
    assert_eq!(cpu.device_id(), None);
    assert_eq!(format!("{}", cpu), "cpu");

    let gpu = Device::Gpu(0);
    assert!(!gpu.is_cpu());
    assert!(gpu.is_gpu());
    assert!(!gpu.is_tpu());
    assert_eq!(gpu.device_id(), Some(0));
    assert_eq!(format!("{}", gpu), "gpu:0");

    let tpu = Device::Tpu(2);
    assert!(!tpu.is_cpu());
    assert!(!tpu.is_gpu());
    assert!(tpu.is_tpu());
    assert_eq!(tpu.device_id(), Some(2));
    assert_eq!(format!("{}", tpu), "tpu:2");
}

#[test]
fn test_shape_abstractions() {
    let s1 = Shape::d1(128);
    assert_eq!(s1.rank(), 1);
    assert_eq!(s1.numel(), 128);
    assert_eq!(s1.dims(), &[128]);

    let s2 = Shape::d2(32, 64);
    assert_eq!(s2.rank(), 2);
    assert_eq!(s2.numel(), 2048);
    assert_eq!(s2.as_2d(), Some((32, 64)));

    let s4 = Shape::d4(4, 3, 32, 32);
    assert_eq!(s4.rank(), 4);
    assert_eq!(s4.numel(), 12288);
    assert_eq!(s4.as_4d(), Some((4, 3, 32, 32)));

    // Tuple conversion
    let from_tuple: Shape = (16, 32).into();
    assert_eq!(from_tuple.as_2d(), Some((16, 32)));
}

#[test]
fn test_tensor_device_methods() {
    let t = Tensor::zeros((4, 8));
    assert_eq!(t.device(), Device::Cpu);
    assert_eq!(t.numel(), 32);
    assert_eq!(t.unified_shape().as_2d(), Some((4, 8)));

    let t_cpu = t.to(Device::Cpu);
    assert_eq!(t_cpu.device(), Device::Cpu);

    let t4 = Tensor::zeros_4d((2, 3, 4, 4));
    assert_eq!(t4.device(), Device::Cpu);
    assert_eq!(t4.numel(), 96);
    assert_eq!(t4.unified_shape().as_4d(), Some((2, 3, 4, 4)));
}

#[test]
fn test_module_device_transfer() {
    let model = Sequential::new(vec![
        Box::new(Linear::new(10, 20)),
        Box::new(ReLu),
        Box::new(Linear::new(20, 5)),
    ]);

    model.to(Device::Cpu);

    for param in model.parameters() {
        assert_eq!(param.device(), Device::Cpu);
    }

    let input = Tensor::random((4, 10));
    let output = model.forward(&input);
    assert_eq!(output.shape(), (4, 5));
    assert_eq!(output.device(), Device::Cpu);
}
