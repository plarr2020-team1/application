# Application
This repository is for testing and comparing different approaches to social distancing.

## Requirements

Clone repository with submodules (`git clone --recurse-submodules ...`).

- **Cuda 10.1**
- **gcc min 7**

Install the submodules in editable mode
```bash
conda create -n socdist-env python=3.7
conda activate socdist-env

conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

pip install -e monoculardepth/monodepth2
pip install -e monoculardepth/mannequinchallenge

pip install -e object-detection-segmentation/yolact 
pip install -e git+https://github.com/CharlesShang/DCNv2@master#egg=dcnv2

pip install -e tracking_wo_bnw

pip install -e human_depth_dataset

pip install -r requirements.txt
```
Retrieve checkpoint for `mannequinchallenge`
```bash
cd monoculardepth/mannequinchallenge && ./fetch_checkpoints.sh && cd ../..
```

Download https://vision.in.tum.de/webshare/u/meinhard/tracking_wo_bnw-output_v2.zip and unzip into `tracking_wo_bnw/output`.

## Usage
```bash
python run.py --video_source samples/mot16.webm --depth_merger median
```

## API Usage
```bash
uvicorn rest:app --reload
curl --location --request POST 'localhost:8000/predict' --form 'file=@example_image.jpg'
```
