import torch
from model import CSIClassifier

def export_to_onnx():
    # 1. Instantiate the model
    num_classes = 5
    model = CSIClassifier(num_classes=num_classes)
    
    # 2. If a trained weights file exists, load it
    try:
        model.load_state_dict(torch.load("csi_model.pth"))
        print("Loaded trained model weights from csi_model.pth")
    except FileNotFoundError:
        print("No trained weights found (csi_model.pth). Exporting model with initialized weights.")
    
    model.eval()
    
    # 3. Create dummy input with the shape: (batch_size, channels, packets, subcarriers)
    # Channel=2 (Amplitude, Phase), packets=10, subcarriers=56
    dummy_input = torch.randn(1, 2, 10, 56, dtype=torch.float32)
    
    # 4. Export the model
    onnx_file_path = "csi_model.onnx"
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_file_path,
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=12,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],     # the model's input names
        output_names=['output'],   # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'packets', 3: 'subcarriers'},  # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model successfully exported to ONNX format at '{onnx_file_path}'!")

if __name__ == '__main__':
    export_to_onnx()
