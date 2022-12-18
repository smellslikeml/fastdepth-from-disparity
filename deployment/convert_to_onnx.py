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
def convert_ONNX(model, save_name="model.onnx", input_shape=[3, 224,224], arch="mobilenet_v2_disp"):

    # set the model to inference mode
    model.eval()

    if arch == 'mobilenet_v2_disp':
        dummy_input_rgb = torch.randn(1, input_shape[0], input_shape[1], input_shape[2], requires_grad=True)
        dummy_input_disp = torch.randn(1, input_shape[0], input_shape[1], input_shape[2], requires_grad=True)
        torch.onnx.export(model,         # model being run
             (dummy_input_rgb, dummy_input_disp),       # model input (or a tuple for multiple inputs)
             save_name,       # where to save the model
             export_params=True,  # store the trained parameter weights inside the model file
             input_names = ['rgb', 'disp'],   # the model's input names
             output_names = ['output'] # the model's output names
             )
    else:
        dummy_input = torch.randn(1, input_shape[0], input_shape[1], input_shape[2], requires_grad=True)
        torch.onnx.export(model,         # model being run
             dummy_input,       # model input (or a tuple for multiple inputs)
             save_name,       # where to save the model
             export_params=True,  # store the trained parameter weights inside the model file
             input_names = ['rgb'],   # the model's input names
             output_names = ['output'] # the model's output names
             )

    print(" ")
    print('Model has been converted to ONNX')

def main():
    global args, best_result, output_directory, train_csv, test_csv
    device = torch.device('cpu')
    arch = args.arch
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
    convert_ONNX(model, save_name=onnx_file, input_shape=input_shape, arch=arch)
    print("\n3. Covert Model to Blob")
    if arch == 'mobilenet_v2_disp':
        shape_settings = "--input_shape={}".format(",".join([str([1] + input_shape), str([1] + input_shape)]))
        input_settings = "--input={}".format("rgb,disp")
    else:
        shape_settings = "--input_shape={}".format(str([1] + input_shape))
        input_settings = "--input={}".format("rgb")
    blob_path = blobconverter.from_onnx(
            model=onnx_file,
            data_type="FP16",
            shaves=6,
            optimizer_params=[input_settings, shape_settings],
        ) 
    print("Saved blob model to {}".format(blob_path))
    print("Done")


if __name__ == "__main__":
    main()
