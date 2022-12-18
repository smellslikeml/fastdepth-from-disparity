FastDepth with Disparity 
============================

This repo contains the code base to train and deploy a modified [FastDepth model architecture](https://arxiv.org/pdf/1709.07492.pdf) using both RBG images and disparity maps inputs for fast depth estimation. We also include scripts for dataset creation and deployment using [OAK-D camera variants](https://shop.luxonis.com/products/oak-d). 

This repo can be used for training and testing of
- Original RGB (or grayscale image) based depth prediction using a MobilenetV2 backbone.
- Our modification using RGB and disparity based depth prediction using a MobilenetV2 backbone.

The original Torch implementation of the paper can be found [here](https://github.com/fangchangma/sparse-to-dense).

## Contents
0. [Requirements](#requirements)
0. [Data](#data)
0. [Training](#training)
0. [Testing](#testing)
0. [Deployment](#deployment)
0. [References](#references)

## Requirements
This code was tested with Python 3, Pytorch 0.4.1, and CUDA 9.2.
We recommend using Docker. Make sure [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) has been properly installed and configured. To build, run:
```
docker build -f docker/Dockerfile -t smellslikeml/fastdepth-disp:latest .
```

To launch a container, run:
```
docker run -it --rm --gpus=all --shm-size=8G --network host -v ${pwd}:/app smellslikeml/fastdepth-disp:latest bash
```

## Data

To train the original model architecture (rgb only input), we recommend to use the NYU Depth V2 dataset as described below. To build a dataset with both RGB and disparity maps, follow the instructions under Custom data.

### NYU Depth V2
Download the preprocessed [NYU Depth V2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and/or [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) dataset in HDF5 formats, and place them under the `data` folder. The downloading process might take an hour or so. The NYU dataset requires 32G of storage space, and KITTI requires 81G.
	```bash
	mkdir data; cd data
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
	tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/kitti.tar.gz
 	tar -xvf kitti.tar.gz && rm -f kitti.tar.gz
	cd ..
	```
### Custom Data
To train with RGB and disparity inputs, use the `extract_data.py` script to create an HDF5 formatted dataset. Ensure that all [depthai](https://docs.luxonis.com/projects/api/en/latest/install/) dependecies are properly installed.

## Training
The training scripts come with several options, which can be listed with the `--help` flag. 
```bash
python3 main.py --help
```

To train the original MobilenetV2-base rgb only input model, run:
```
python3 main.py -a mobilenet_v2 -c l1 -d deconv3 -m rgb --lr 0.01 -s 100 --data nyudepthv2 --epochs 20
```

To train the modified MobilenetV2-base, rgb and disparity map input model, run:
```
python3 main.py -a mobilenet_v2_disp -c l1 -d deconv3 -m rgb --lr 0.01 -s 100 --data nyudepthv2_disp --epochs 20
```

Training results will be saved under the `results` folder. To resume a previous training, run
```bash
python3 main.py --resume [path_to_previous_model]
```

## Testing
To test the performance of a trained model without training, simply run main.py with the `-e` option. For instance,
```bash
python3 main.py --evaluate [path_to_trained_model]
```

## Deployment

A trained model checkpoint can be converted and deployed for inference. The original pytorch checkpoints need to be converted to `.blob` format to run on Oak-D cameras.

### Model conversion
Covert a checkpoint model to blob using the following command:
```
python3 convert_to_onnx.py --evaluate results/path-to/model_best.pth.tar --input-shape 3 224 224 --onnx-file mobilenetv2_disp.onnx --arch mobilenet_v2_disp
```

Use the `run_inference.py` example within `deployment/`. Make sure to point to the newly created blob model:
```
python3 deployment/run_inference.py -nn <path-to-blob-model> -shape 224x224
```

## References

@article{Ma2017SparseToDense,
	title={Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image},
	author={Ma, Fangchang and Karaman, Sertac},
	booktitle={ICRA},
	year={2018}
}
@article{ma2018self,
	title={Self-supervised Sparse-to-Dense: Self-supervised Depth Completion from LiDAR and Monocular Camera},
	author={Ma, Fangchang and Cavalheiro, Guilherme Venturelli and Karaman, Sertac},
	journal={arXiv preprint arXiv:1807.00275},
	year={2018}
}

Please create a new issue for code-related questions. Pull requests are welcome.
