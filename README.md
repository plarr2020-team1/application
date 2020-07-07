# Application
This repository is for testing and comparing different approaches to social distancing.

## Requirements

Clone repository with submodules (`git clone --recurse-submodules ...`).

Install the submodules in editable mode
```bash
pip install -e monoculardepth/monodepth2

pip install -e object-detection-segmentation/yolact 
pip install -e git+https://github.com/CharlesShang/DCNv2@master#egg=dcnv2
```

## Usage
```bash
python run.py --video_source samples/mot16.webm
```

## API Usage
```bash
pip install -r requirements.txt
uvicorn rest:app --reload
curl --location --request POST 'localhost:8000/predict' --form 'file=@example_image.jpg'
```