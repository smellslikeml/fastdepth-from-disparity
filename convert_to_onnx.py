import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.onnx
cudnn.benchmark = True

from models import ResNet
from mobilenet_models import MobileNetSkipAdd
from metrics import AverageMeter, Result
from dataloaders.dense_to_sparse import UniformSampling, SimulatedStereo
import criteria
import utils

args = utils.parse_command()
print(args)

#Function to Convert to ONNX
def convert_ONNX(model, save_name="model_nyu.onnx", input_size=(3, 480, 640)):

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

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
    print("\n2. Covert Model")
    convert_ONNX(model, save_name="nyudepthv2_mobilenetv2_lr001.onnx")
    print("Done")


if __name__ == "__main__":
    main()
