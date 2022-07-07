import cv2
import numpy as np


#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))




net = cv2.dnn.readNetFromDarknet('E:\\master_thesis\\project\\vehicle_dataset\\model_test\\yolov3_custom1.cfg',
                                 "E:\\master_thesis\\project\\vehicle_dataset\\yolov3_model\\yolov3_final.weights")
classes = ['car', 'truck', 'bicycle', 'bus', 'pedestrian', 'motor_bike']
# classes = []
# with open("E:\\master_thesis\\project\\car_yolov3\\yolo_pretrained\\coco.names", 'r') as f:
# classes = [line.strip() for line in f.readlines()]
cap = cv2.VideoCapture("E:\\master_thesis\\project\\vehicle_dataset\\model_test\\3.webm")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
out = cv2.VideoWriter('E:\\master_thesis\\project\\vehicle_dataset\\model_test\\yolo_output.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
while cap.isOpened():
    rate, frame = cap.read()
    img = frame
    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    my_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # my_img = img
    ht, wt, _ = my_img.shape

    blob = cv2.dnn.blobFromImage(my_img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    blob.shape

    net.setInput(blob)
    last_layer = net.getUnconnectedOutLayersNames()

    layer_out = net.forward(last_layer)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_out:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > .6:
                center_x = int(detection[0] * wt)
                center_y = int(detection[1] * ht)
                w = int(detection[2] * wt)
                h = int(detection[3] * ht)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(my_img, (x, y), (x + w, y + h), color, 2)

        y_img = cv2.resize(my_img, size, interpolation=cv2.INTER_AREA)
        out.write(y_img)
        cv2.imshow('img', my_img)
        # cv2.resizeWindow('img', 800, 800)
        # cv2.imshow("img", cv2.resize(my_img, (0, 0), fx=0.2, fy=0.2))
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
out.release()
cv2.destroyAllWindows()
