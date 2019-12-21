import time
import random
import os
from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
from PIL import Image
import numpy as np

def yolo(input):
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
    x, img = data.transforms.presets.yolo.load_test(input, short=512)
    box_ids, scores, bboxes = net(x)
    ax = utils.viz.cv_plot_bbox(img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

def frcnn(input):
    net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
    x, orig_img = data.transforms.presets.rcnn.load_test(input)
    box_ids, scores, bboxes = net(x)
    ax = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)

def getTimeYOLO(arr):
    start = time.time()
    for val in arr:
        yolo(val)
    return time.time() - start

def getTimeFRCNN(arr):
    start = time.time()
    for val in arr:
        frcnn(val)
    return time.time() - start

inputDir='inputBenchmark'
testNum = 5
quantity = 3

print('Going to test %s tests. Each test will randomly choose %s images to process' % (str(testNum), str(quantity)))
for root, dirs, files in os.walk(inputDir):
    for i in range(testNum):
        arr = []
        for j in range(quantity):
            r = os.path.join(inputDir, random.choice(files))
            arr.append(r)
        print('Test number %s' % (str(i)))
        print('Running YOLO')
        y = getTimeYOLO(arr)
        print('Running Faster RCNN')
        f = getTimeFRCNN(arr)
        print('Yolo processed %s' % (y))
        print('Faster RCNN processed %s' % (f))
        if f > y:
            print('Yolo wins in test number %s: %s seconds' % (str(i), f-y))
        else:
            print('Faster RCNN wins in test number %s: %s seconds' % (str(i), y-f))

        print('================')