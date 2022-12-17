import os
import cv2
import numpy as np
import blobconverter

import torch
from torch.utils.data import DataLoader

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.onnx
cudnn.benchmark = True

import utils

args = utils.parse_command()
print(args)

#Function to Convert to ONNX
def convert_ONNX(model, save_name="model.onnx", input_shape=[3, 224,224]):

    # set the model to inference mode
    model.eval()

    # Create a dummy input tensor
    if len(input_shape) == 3:
        dummy_input = torch.randn(1, input_shape[0], input_shape[1], input_shape[2], requires_grad=True)
    elif len(input_shape) == 4:
        dummy_input = torch.randn(1, input_shape[0], input_shape[1], input_shape[2], input_shape[3], requires_grad=True)
    else:
        raise Exception('Unsupported input shape. Must be of length 3 or 4')

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         save_name,       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'] # the model's output names
         )
    print(" ")
    print('Model has been converted to ONNX')

def main():
    global args, best_result, output_directory, train_csv, test_csv
    device = torch.device('cpu')
    input_shape = args.input_shape
    onnx_file = args.onnx_file

    print("\n1. Define Model")
    assert os.path.isfile(args.evaluate), \
    "=> no best model found at '{}'".format(args.evaluate)
    print("=> loading best model '{}'".format(args.evaluate))
    checkpoint = torch.load(args.evaluate, map_location=device)
    output_directory = os.path.dirname(args.evaluate)
    args = checkpoint['args']
    start_epoch = checkpoint['epoch'] + 1
    best_result = checkpoint['best_result']
    model = checkpoint['model'].to(device)
    print("\n2. Covert Model to ONNX")
    convert_ONNX(model, save_name=onnx_file, input_shape=input_shape)
    print("\n3. Covert Model to Blob")
    shape_settings = "--input_shape={}".format(str([1] + input_shape))
    blob_path = blobconverter.from_onnx(
            model=onnx_file,
            data_type="FP16",
            shaves=6,
            optimizer_params=[shape_settings],
        ) 
    print("Saved blob model to {}".format(blob_path))
    print("Done")


if __name__ == "__main__":
    main()
