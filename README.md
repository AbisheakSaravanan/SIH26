# SIH26

The Individual repo for SIH Internal Hackathon 2026

Created on 23rd August, Abisheak Saravanan.

# Wi-Fi Object Sensing

This project explores the possibility of using Wi-Fi signals to detect and understand the presence, movement, and basic structure of objects.

The goal is to investigate whether Wi-Fi can be used for sensing in a way similar to sonar or radar, with potential applications in object detection, indoor monitoring, and wireless environmental awareness.

This project is currently in the early exploration and development stage.

# Working Model Diagram

<img width="1280" height="853" alt="image" src="https://github.com/user-attachments/assets/75834f7b-0421-44e8-936e-b5bbd6abee8f" />

## ONNX Model Specifications

The network is compiled to ONNX format with dynamic shapes supporting variable batch sizes and subcarrier counts.

* **Model File**: `csi_model.onnx` (Opset 18)
* **Input Tensor (`input`)**: `[batch_size, 2, packets, subcarriers]` (Float32)
* **Output Tensor (`output`)**: `[batch_size, 5]` (Float32 class logits)

### Target Classification Classes
1. `Standing` (Index 0)
2. `Sitting` (Index 1)
3. `Walking` (Index 2)
4. `Running` (Index 3)
5. `Falling` (Index 4)

*For the full architecture breakdown, layer parameters, and runtime inference code example, see [onnx_details.md](./onnx_details.md).*
