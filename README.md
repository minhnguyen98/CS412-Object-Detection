# CS412-Object-Detection
Object detection in class CS412 of APCS

# How to run

## Initialize project

```bash
bash init.sh
sudo apt install python-tk
pip3 install --upgrade mxnet-cu90mkl gluoncv
sudo apt install python-opencv
```

## run YOLOv3

Just import images to inputYOLOv3/ directory and:

```bash
python3 runYOLOv3.py
```

## Faster RCNN

Just import images to inputFasterRCNN/ directory and:

```bash
python3 runFasterRCNN.py
```