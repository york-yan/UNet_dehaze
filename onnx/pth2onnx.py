import torch
import sys
sys.path.append('../')
from model import unet

def export_onnx_model(model,dummy_input,onnx_output_path,input_names=['input'],output_names=['output'],dynamic_axes=None):
    """
    Exports the given PyTorch model to an ONNX file.
    
    Args:
        model (torch.nn.Module): The PyTorch model to be exported.
        dummy_input (torch.Tensor): A dummy input tensor that matches the shape and dtype of the model's input.
        onnx_output_path (str): The path where the ONNX file will be saved.
        input_names (List[str], optional): The names of the model's input tensors. Defaults to ['input'].
        output_names (List[str], optional): The names of the model's output tensors. Defaults to ['output'].
        dynamic_axes (Dict[str, Dict[int, str]], optional): A dictionary specifying the dynamic axes of the model's input tensors. Defaults to None.
    """
    torch.onnx.export(model,dummy_input,onnx_output_path,input_names=input_names,output_names=output_names,dynamic_axes=dynamic_axes)


model=unet.UNet()

model.load_state_dict(torch.load('../weight/unet.pth'))
model.eval()
dummy_input=torch.randn(1,3,512,512)
onnx_output_path='./unet_dynamic.onnx'
dynamic_axes={'input':{0:'batch',2:'height',3:'width'},'output':{0:'batch',2:'height',3:'width'}}
export_onnx_model(model,dummy_input,onnx_output_path,dynamic_axes=dynamic_axes)