#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np
from depthai_sdk import FPSHandler
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--nn_path', type=str, help="select model blob path for inference", required=True)
parser.add_argument('-shape', '--shape', type=str, help="model input shape, same as used blob", choices=["224x224"], required=True)

args = parser.parse_args()

NN_PATH = args.nn_path
TARGET_SHAPE = 224, 224
NN_SHAPE = [int(dim) for dim in args.shape.split("x")]

extended_disparity = True
subpixel = False
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

# Define sources and outputs
rgb = pipeline.create(dai.node.MonoCamera)
disp = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)

# Properties
rgb.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
rgb.setBoardSocket(dai.CameraBoardSocket.LEFT)

disp.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

disp.setBoardSocket(dai.CameraBoardSocket.RIGHT)

depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

# NN
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(NN_PATH)

rgb.out.link(depth.left)
disp.out.link(depth.right)

# Resize to NN shape
manipRgb = pipeline.create(dai.node.ImageManip)
manipRgb.initialConfig.setResize(NN_SHAPE[1], NN_SHAPE[0])
manipRgb.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
depth.rectifiedLeft.link(manipRgb.inputImage)

manipDisp = pipeline.create(dai.node.ImageManip)
manipDisp.initialConfig.setResize(NN_SHAPE[1], NN_SHAPE[0])
manipDisp.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
depth.rectifiedRight.link(manipDisp.inputImage)

# Set NN left/right inputs
manipRgb.out.link(nn.inputs["rgb"])
manipDisp.out.link(nn.inputs["disp"])

# NN and stereoDepth outputs
nn_xout = pipeline.createXLinkOut()
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)

#xout = pipeline.create(dai.node.XLinkOut)
#xout.setStreamName("disparity")
#depth.disparity.link(xout.input)
depth.disparity.link(manipDisp.inputImage)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    qNn = device.getOutputQueue(name="nn", maxSize=2, blocking=False)
    #qDisp = device.getOutputQueue(name="disparity", maxSize=2, blocking=False)

    fps_handler = FPSHandler()

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255,0,255)
    thickness = 1

    model_max_disp = 192
    nn_disp_multiplier =  255.0 / model_max_disp 
    scale_multiplier = TARGET_SHAPE[1] / NN_SHAPE[1]

    stereo_disp_multiplier = 255.0 / depth.initialConfig.getMaxDisparity()

    while True:
        
        fps_handler.tick("nn")
        fps = fps_handler.tickFps("nn")

        nn_output = np.array(qNn.get().getData()).view(np.float16)
        #disp_output = qDisp.get().getFrame()

        nn_output = nn_output.reshape((1, NN_SHAPE[0], NN_SHAPE[1])).squeeze()
        nn_disp = nn_output[0, :].astype("uint8")

        nn_disp = (nn_disp * nn_disp_multiplier * scale_multiplier).astype(np.uint8)
        nn_disp_vis = cv2.resize(nn_disp, (TARGET_SHAPE[1], TARGET_SHAPE[0]))
        #nn_disp_vis = cv2.applyColorMap(nn_disp, cv2.COLORMAP_INFERNO)

        #stereo_disp = (disp_output * stereo_disp_multiplier).astype(np.uint8)
        #stereo_disp_vis = cv2.applyColorMap(stereo_disp, cv2.COLORMAP_INFERNO)

        nn_disp_vis = cv2.putText(nn_disp_vis, f"NN FPS {fps:.2f}", (20, 20), font, fontScale, color, thickness, cv2.LINE_AA)
        #stereo_disp_vis = cv2.putText(stereo_disp_vis, f"Stereo Disp", (20, 20), font, fontScale, color, thickness, cv2.LINE_AA)
        
        #vis = np.concatenate(
        #    [nn_disp_vis, stereo_disp_vis],
        #    axis=1
        #)
        vis = nn_disp_vis
        cv2.imshow("Disparity", vis)
        
        if cv2.waitKey(1) == ord('q'):
            break
