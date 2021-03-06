# CS412-Object-Detection
Object detection in class CS412 of APCS

# How to run

## Initialize project

```bash
bash init.sh
sudo apt install python-tk
pip install gluoncv
pip install --upgrade mxnet gluoncv
pip install opencv-python
```

## run YOLOv3

Just import images to inputYOLOv3/ directory and:

```bash
python3 runYOLOv3.py
```

The result images will be in outputYOLOv3/

## Faster RCNN

Just import images to inputFasterRCNN/ directory and:

```bash
python3 runFasterRCNN.py
```

The result images will be in outputFasterRCNN/

# Some example after running detection

## Image 1

Original:

![Original 1](https://github.com/phvietan/CS412-Object-Detection/blob/master/example/original1.jpg)

After dectection:

![Detected 1](https://github.com/phvietan/CS412-Object-Detection/blob/master/example/detected1.jpg)

## Image 2

Original:

![Original 2](https://github.com/phvietan/CS412-Object-Detection/blob/master/example/original2.jpg)

After dectection:

![Detected 2](https://github.com/phvietan/CS412-Object-Detection/blob/master/example/detected2.jpg)

## Image 3

Original:

![Original 3](https://github.com/phvietan/CS412-Object-Detection/blob/master/example/original3.jpg)

After dectection:

![Detected 3](https://github.com/phvietan/CS412-Object-Detection/blob/master/example/detected3.jpg)

## Image 4

Original:

![Original 4](https://github.com/phvietan/CS412-Object-Detection/blob/master/example/original4.jpg)

After dectection:

![Detected 4](https://github.com/phvietan/CS412-Object-Detection/blob/master/example/detected4.jpg)
