# Models-Manager-Pro
Models Manager Pro v4.0 (A.I. Copilot Edition) is a unified desktop platform designed for the complete management of YOLO (v5–v12) and CNN (torchvision) computer vision models
. Built using the PySide6 framework, the application provides a modern graphical interface for handling the entire machine learning pipeline, from training and optimization to live deployment and statistical analysis
.
1. Comprehensive Model Support
The application supports a wide range of architectures:
YOLO Models: Full integration with the Ultralytics library for Object Detection and Classification, supporting versions from v5 up to v12
.
CNN Classifiers: Support for torchvision architectures, specifically MobileNet V2/V3 (Small and Large) and ResNet-50/101, using a native PyTorch training loop
.
2. Advanced Training & Optimization
Training is highly customizable, allowing users to configure epochs, batch size, image size, and specific optimizers like Adam, AdamW, and SGD
.
Triton Acceleration: For YOLO models on GPU, the app supports TorchInductor/Triton compilation with modes like "reduce-overhead" and "max-autotune" to significantly speed up training
.
A.I. Copilot: An intelligent assistant powered by the Groq LLM API analyzes training logs and hardware environment to suggest optimized hyperparameters
.
Dataset Management: It includes automatic dataset scanning and supports both YAML-based detection datasets and ImageFolder structures for classification
.
3. Multi-Backend Exporting
Models can be exported into various high-performance formats to suit different deployment needs:
ONNX: Industry-standard format for cross-platform inference
.
TensorRT (.engine): Optimized for NVIDIA GPUs with built-in signature management for reliable caching and validation
.
NCNN: Specifically for mobile and edge device deployment
.
4. Real-time Inference & Benchmarking
Live Camera Detection: Provides real-time inference from camera streams with specialized overlays, including confidence bars and rank badges for CNN classifications
.
Video File Inference: Allows frame-by-frame processing of video files (mp4, avi, etc.) with the option to save annotated outputs
.
Automated Benchmarking: Includes dedicated tools to measure and compare the FPS and latency of different backends (PyTorch, ONNX, TensorRT, NCNN) on the user's specific hardware
.
5. Evaluation & Comparative Analysis
Professional PDF Reports: Automatically generates detailed training and detection reports featuring Matplotlib charts for loss curves, accuracy/mAP progression, and class distribution
.
Comparison Dialog: A sophisticated analysis tool that compares multiple training runs using radar charts and ranking systems to recommend the "Best Model" based on seven distinct performance criteria
.
6. System Stability & Diagnostics
Resource Monitoring: Real-time tracking of CPU, RAM, and GPU usage
.
Robust Logging: Features advanced error handling, crash logs with full thread dumps, and a Python faulthandler for capturing low-level system crashes
.
UI Features: Supports Light/Dark theme toggling and adaptive UI scaling for different screen resolutions

