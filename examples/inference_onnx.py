#!/usr/bin/env python3
"""
Inference Script for ONNX models exported by Rensor (NeuralFlow).

Features:
- Validates the ONNX graph using onnx.checker
- Inspects input/output shapes, tensor types, and opset metadata
- Loads test samples from the raw MNIST binary dataset (data/mnist)
- Runs batch and single-sample inference using onnxruntime
- Measures inference latency and throughput (img/s)
"""

import os
import sys
import time
import struct
import argparse
import numpy as np
import onnx
import onnxruntime as ort


def load_mnist_test_samples(image_path: str, label_path: str, num_samples: int = 10):
    """Loads raw MNIST IDX test samples and normalizes pixels to [0.0, 1.0]."""
    if not (os.path.exists(image_path) and os.path.exists(label_path)):
        return None, None

    with open(image_path, "rb") as f_img, open(label_path, "rb") as f_lbl:
        _, total_imgs, rows, cols = struct.unpack(">IIII", f_img.read(16))
        _, total_lbls = struct.unpack(">II", f_lbl.read(8))
        assert total_imgs == total_lbls, "Image and label count mismatch"

        count = min(num_samples, total_imgs)
        image_pixels = rows * cols  # 784

        raw_images = np.frombuffer(f_img.read(count * image_pixels), dtype=np.uint8)
        images = raw_images.reshape(count, 1, rows, cols).astype(np.float32) / 255.0

        raw_labels = np.frombuffer(f_lbl.read(count), dtype=np.uint8)
        labels = raw_labels.astype(np.int64)

    return images, labels


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description="Run ONNX inference on Rensor models")
    parser.add_argument(
        "--model",
        type=str,
        default="examples/CNN.onnx",
        help="Path to exported ONNX model (default: examples/CNN.onnx)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of MNIST test samples to evaluate (default: 10)",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=1000,
        help="Number of iterations for latency benchmarking (default: 1000)",
    )
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: ONNX model not found at '{model_path}'.")
        print("Please train and export the model first with:")
        print("  cargo run --example CNN --release")
        sys.exit(1)

    print("=" * 65)
    print("  Rensor - ONNX Inference & Validation Engine")
    print("=" * 65)
    print(f"Model file: {model_path} ({os.path.getsize(model_path):,} bytes)\n")

    # 1. Inspect and Validate Model using ONNX
    print("[1/4] Inspecting ONNX Graph Structure...")
    onnx_model = onnx.load(model_path)
    print(f"  Producer Name  : {onnx_model.producer_name}")
    print(f"  Producer Ver   : {onnx_model.producer_version}")
    print(f"  IR Version     : {onnx_model.ir_version}")
    opset = onnx_model.opset_import[0].version if onnx_model.opset_import else "N/A"
    print(f"  Opset Version  : {opset}")

    try:
        onnx.checker.check_model(onnx_model)
        print("  Graph Checker  : VALID (no structural or shape errors)")
    except Exception as e:
        print(f"  Graph Checker  : WARNING: {e}")

    # 2. Initialize ONNX Runtime Session
    print("\n[2/4] Initializing ONNX Runtime Inference Session...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options=opts)

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    print(f"  Input Name     : '{input_meta.name}' | Shape: {input_meta.shape} | Type: {input_meta.type}")
    print(f"  Output Name    : '{output_meta.name}' | Shape: {output_meta.shape} | Type: {output_meta.type}")

    # 3. Run Inference on Real MNIST Test Data
    print("\n[3/4] Running Inference on MNIST Test Samples...")
    test_images_path = "data/mnist/t10k-images-idx3-ubyte"
    test_labels_path = "data/mnist/t10k-labels-idx1-ubyte"

    images, labels = load_mnist_test_samples(test_images_path, test_labels_path, args.num_samples)

    if images is not None:
        # Run inference on batch
        outputs = session.run([output_meta.name], {input_meta.name: images})[0]
        probs = softmax(outputs)
        predictions = np.argmax(probs, axis=1)

        print(f"{'-' * 65}")
        print(f"{'Sample':<8} | {'Ground Truth':<14} | {'Predicted':<10} | {'Confidence':<12} | {'Status'}")
        print(f"{'-' * 65}")

        correct = 0
        for i in range(len(labels)):
            gt = labels[i]
            pred = predictions[i]
            conf = probs[i][pred] * 100.0
            is_correct = (gt == pred)
            if is_correct:
                correct += 1
            status = "MATCH" if is_correct else "MISMATCH"
            print(f"{i:<8} | {gt:<14} | {pred:<10} | {conf:>7.2f}%    | {status}")

        print(f"{'-' * 65}")
        accuracy = (correct / len(labels)) * 100.0
        print(f"Sample Batch Accuracy: {accuracy:.1f}% ({correct}/{len(labels)})\n")
    else:
        print("  MNIST binary test files not found in data/mnist/.")
        print("  Falling back to synthetic tensor inference (1, 1, 28, 28)...")
        synthetic_input = np.random.randn(1, 1, 28, 28).astype(np.float32)
        outputs = session.run([output_meta.name], {input_meta.name: synthetic_input})[0]
        probs = softmax(outputs)[0]
        pred_class = np.argmax(probs)
        print(f"  Predicted class: {pred_class} (confidence: {probs[pred_class]*100:.2f}%)\n")

    # 4. Latency & Throughput Benchmark
    print(f"[4/4] Benchmarking Inference Performance ({args.benchmark_iters} iterations)...")
    single_img = (images[0:1] if images is not None else np.random.randn(1, 1, 28, 28).astype(np.float32))

    # Warmup
    for _ in range(50):
        _ = session.run([output_meta.name], {input_meta.name: single_img})

    # Timed run
    start_time = time.perf_counter()
    for _ in range(args.benchmark_iters):
        _ = session.run([output_meta.name], {input_meta.name: single_img})
    total_time = time.perf_counter() - start_time

    avg_latency_ms = (total_time / args.benchmark_iters) * 1000.0
    throughput = args.benchmark_iters / total_time

    print(f"  Batch size       : 1")
    print(f"  Average Latency  : {avg_latency_ms:.3f} ms / image")
    print(f"  Throughput       : {throughput:,.1f} inferences / sec")
    print("=" * 65)
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()

