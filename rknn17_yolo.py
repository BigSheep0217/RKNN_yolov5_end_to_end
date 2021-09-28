import time
import numpy as np
import torchvision
import torch
import cv2
from rknn.api import RKNN

labels = ['blue', 'green', 'yellow', 'ass']

if __name__ == "__main__":

    img_path = "./2.jpg"
    img_h, img_w = 288, 288

    rknn = RKNN()
    ret = rknn.load_rknn('./plate.rknn')

    # Set inputs
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_w, img_h))
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
# #############################################################
# ####################### Inference ###########################
# #############################################################

    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    print('done')
    nout = len(outputs)
    print("output branch : ", nout)
    print(outputs[0].shape)
    data = outputs[0][0]
    res, ch = data.shape
# #############################################################
# ######################### explain ###########################
# #############################################################
    boxes = []
    conf = []
    labels = []
    for i in range(res):
        xywhpc = data[i, :]
        if xywhpc[4] > 0.01:
            # print("get target : ", xywhpc)
            boxes.append([int(xywhpc[0]-xywhpc[2]/2), int(xywhpc[1]-xywhpc[3]/2), int(xywhpc[0]+xywhpc[2]/2), int(xywhpc[1]+xywhpc[3]/2)])
            # boxes.append(xywhpc[:4])
            conf.append(xywhpc[4])
            labels.append(np.argmax(xywhpc[5:]))
            print(np.argmax(xywhpc[5:]), xywhpc[4], ([int(xywhpc[0]-xywhpc[2]/2), int(xywhpc[1]-xywhpc[3]/2), int(xywhpc[0]+xywhpc[2]/2), int(xywhpc[1]+xywhpc[3]/2)]))


# #############################################################
# ########################### NMS #############################
# #############################################################

    if len(boxes) != 0:
        box_index = torchvision.ops.nms(torch.tensor(np.array(
            boxes, float)), torch.tensor(np.array(conf, float)), float(0.5))  # NMS
        for i in box_index:
            print(labels[i], conf[i], boxes[i])
            # cv2.rectangle(img, (int(boxes[i][0]-boxes[i][2]/2), int(boxes[i][1]-boxes[i][3]/2)), (int(boxes[i][0]+boxes[i][2]/2), int(boxes[i][1]+boxes[i][3]/2)), (255, 255, 255))
            cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (255, 255, 255))
    cv2.imwrite("./result.jpg", img)



