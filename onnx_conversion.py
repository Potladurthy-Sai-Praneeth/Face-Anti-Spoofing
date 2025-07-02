import argparse
import onnx
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, create_calibrator, CalibrationMethod, CalibrationDataReader
import os
import numpy as np
from depth_map_model import FineTuneDepthAnything
from classifier import DepthClassifier
import torch
from config import CONFIG
import multiprocessing
from dataloader import CustomDataset, collate_fn
from torch.utils.data import DataLoader, DistributedSampler
from anti_spoofing import AntiSpoofing


def export_to_onnx(model,device,filename='depth_anything_onnx.onnx',save_path='/kaggle/working/',image_size=CONFIG['img_size']):
    dummy_ip = torch.randn(1,3,image_size[0],image_size[1],device=device)
    torch.onnx.export(
        model,
        dummy_ip,
        os.path.join(save_path,filename),
        export_params=True,
        input_names=["input"],
        output_names=["predicted_depth", "binary_predictions"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch_size"}, "predicted_depth": {0: "batch_size"}, "binary_predictions": {0: "batch_size"}},
        # dynamo=True
    )

    print(f'Model exported to onnx format and saved at {os.path.join(save_path,filename)}')

class CalibrationDataset(CalibrationDataReader):
    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader
        self.enum_data = iter(dataloader)

    def to_numpy(self, pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:
            inputs, binary_labels,gt = batch
            return {'input': self.to_numpy(inputs)}
        else:
          return None

    def rewind(self):
        self.enum_data = iter(self.dataloader)


def quantize_onnx_model(onnx_model_path,dataloader,device,filename='quanztied_depth_anything.onnx',path='/kaggle/working/',quantization_type='static'):

    calibration = CalibrationDataset(dataloader)

    if quantization_type == 'static':
        quantize_static(
            model_input=onnx_model_path,
            model_output=os.path.join(path,filename),
            calibration_data_reader=calibration,
            quant_format=QuantFormat.QDQ, 
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            use_external_data_format=False,
        )

    else:
        quantize_dynamic(
            model_input=onnx_model_path,
            model_output=os.path.join(path,filename),
            weight_type=QuantType.QInt8,
            # optimize_model=False,
            extra_options={'ActivationSymmetric': True}
        )

    print(f'Model exported to onnx format and saved at {os.path.join(path,filename)}')
    

def create_inference_session(quantized_model_path):
    """Create ONNX Runtime inference session"""
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = multiprocesssing.cpu_count()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Create inference session
    session = onnxruntime.InferenceSession(
        quantized_model_path,
        options,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    return session


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX Conversion and Quantization')
    parser.add_argument('--quantization_type', type=str, choices=['static', 'dynamic'], default='static', help='Type of quantization to apply (static or dynamic)')
    parser.add_argument('--depth_model_path', type=str, default='/kaggle/working', help='Path to the Depth Anything torch trained model')
    parser.add_argument('--classifier_path', type=str, default='/kaggle/working', help='Path to the Classifier torch trained model')
    parser.add_argument('--onnx_model_name', type=str, default='depth_anything_onnx.onnx', help='Name of the ONNX model to be saved')
    parser.add_argument('--save_path', type=str, default='/kaggle/working/')
    parser.add_argument('--val_path', type=str, default='/kaggle/input/anti-spoofing-dataset/validation', help='Path to the validation dataset for calibration')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AntiSpoofing(depth_model_path= args.depth_model_path, classifier_path=args.classifier_path).to(device)
    model.eval()

    export_to_onnx(model,device,filename=args.onnx_model_name,save_path=args.save_path,image_size=CONFIG['img_size'])
    onnx_model_path = os.path.join(args.save_path, args.onnx_model_name)

    val_dataset = CustomDataset(args.val_path,img_size=CONFIG['img_size'])
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True, 
        num_workers=multiprocessing.cpu_count(), 
        pin_memory=True,
        collate_fn=collate_fn,
    )

    quantize_onnx_model(onnx_model_path,val_loader,device,filename=f'quantized_{args.onnx_model_name}',path='/kaggle/working/',quantization_type=args.quantization_type)

    print("ONNX conversion and quantization completed.")
