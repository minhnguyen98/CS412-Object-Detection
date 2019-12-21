import time
import os
from matplotlib import pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
from PIL import Image
import numpy as np

def processImg(input, output):
    net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)
    x, orig_img = data.transforms.presets.rcnn.load_test(input)
    box_ids, scores, bboxes = net(x)
    ax = utils.viz.cv_plot_bbox(orig_img, bboxes[0], scores[0], box_ids[0], class_names=net.classes)
    img = Image.fromarray(ax, 'RGB')
    img.save(output)

start_time = time.time()
inputDir = 'inputFasterRCNN'
outputDir = 'outputFasterRCNN'

for root, dirs, files in os.walk(inputDir):
    for filename in files:
        input = os.path.join(inputDir, filename)
        output = os.path.join(outputDir, filename)
        cur = time.time()
        processImg(input, output)
        print("Processed %s costs: %s seconds ---" % (input, time.time() - cur))

print("Total time it took %s" % (time.time() - start_time))