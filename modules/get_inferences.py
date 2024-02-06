import cv2
import numpy as np
import torch
from torchvision.ops import nms

# weights_file = "weights/yolov4-tiny.weights"
# cfg_file = "cfg/yolov4-tiny.cfg"

weights_file = "../yolov4.weights"
cfg_file = "../yolov4.cfg"

# load yolo
net = cv2.dnn.readNet(weights_file, cfg_file)
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# use cv2 to make inferences on image
def infer_image(image_path, classes):
    # read image
    img = cv2.imread(image_path)

    # no need to get channels so use _
    height, width, _ = img.shape

    # detect objects and get network output layers
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # get detections
    detections = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # if class_id = "dont_show", do not append
                if classes[class_id] == "dont_show":
                    continue

                detections.append([x, y, w, h, class_id, confidence])

    # check if there are no detections
    if len(detections) == 0:
        return torch.tensor([])

    detections = torch.tensor(detections)

    # convert bounding boxes from (x, y, w, h) to (x1, y1, x2, y2)
    bbs = detections[:, :4].clone()
    bbs[:, 2] = bbs[:, 0] + bbs[:, 2]  # x2 = x + w
    bbs[:, 3] = bbs[:, 1] + bbs[:, 3]  # y2 = y + h
    
    # apply non-max suppression
    confidences = detections[:, 5]
    keep = nms(bbs, confidences, iou_threshold=0.5)

    # keep only non-overlapping detections
    nms_detections=detections[keep]

    # return detections after NMS as tensor
    return nms_detections