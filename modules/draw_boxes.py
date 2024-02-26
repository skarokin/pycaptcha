import cv2
import numpy as np
import torch

def draw_boxes(image_path, classes, tensor):
    # when street view api works, no longer needs this line :D
    img = cv2.imread(image_path)

    # get detections as a tensor
    detections = tensor

    # if no detections, just show image
    if detections.shape[0] == 0:
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # convert tensor to numpy array and extract boxes, class ids, and confidences
    detections.numpy()

    bbs = detections[:, :4].tolist()
    class_ids = detections[:, 4].to(torch.int).tolist()
    confidences = detections[:, 5].tolist()

    # go through detections and draw boxes
    for i in range(len(detections)):
        if isinstance(i, (list, np.ndarray)) and len(i) > 0:
            i = i[0]
        bb = bbs[i]
        x = bb[0]
        y = bb[1]

        label = str(classes[class_ids[i]]) + " " + str(round(confidences[i], 2))

        # if label is dont_show, do not draw
        if classes[class_ids[i]] == "dont_show":
            continue
        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])), (0, 255, 0), 2)
        x = int(bb[0])
        y = int(bb[1])
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


