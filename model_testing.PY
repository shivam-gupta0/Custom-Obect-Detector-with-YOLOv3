import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

net = cv2.dnn.readNetFromDarknet('E:\\project\\vehicle_dataset\\model_test\\yolov3_custom1.cfg', "E:\\project\\vehicle_dataset\\yolov3_model\\12000_epc\\yolov3_final.weights")
print(net)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
classes = ['car', 'truck', 'bicycle', 'bus', 'pedestrian', 'motor_bike']


#images_path = glob.glob("E:\\project\\vehicle_dataset\\HH_images\*.jpg")
images_path = glob.glob("E:\\project\\vehicle_dataset\\new_images\\test_data\*.jpg")
#print(images_path[0])
for img_path in images_path:
    img1 = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.flip(img1, 1)
    scale_percent = 100  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    my_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    #my_img = img
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
            cv2.putText(my_img, label + " " + confidence, (x, y + 2), font, 1, (255, 255, 0), 2)

        cv2.imshow('img', my_img)
        #cv2.resizeWindow('img', 800, 800)
        #cv2.imshow("img", cv2.resize(my_img, (0, 0), fx=0.2, fy=0.2))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
