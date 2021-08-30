import cv2
import numpy as np
import math
min_confidence = 0.5
people_location = []
# Load Yolo
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
classes = []
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

whitebackground = 'output/whitebackground.png'
white = cv2.imread(whitebackground)
whiteh, whitew, whitec = white.shape

model_name = 'res10_300x300_ssd_iter_140000.caffemodel'
prototxt_name = 'deploy.prototxt.txt'
min_confidence = 0.6


file_name = "output/recommand_picture.png"
img = cv2.imread(file_name)
height, width, channel = img.shape
# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)


face_list = []
face_size = []
need_change = []

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > min_confidence:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        if(classes[class_ids[i]] == 'person'):
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            #print(i, label)
            #color = colors[i]
            #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
           # cv2.putText(img, label, (x, y + 30), font, 2, (0, 255, 0), 1)
            people_location.append([x, w, y, h])
            
h, w, c = img.shape
width_20 = w * 0.08
height_20 = h * 0.08

list_a=[]
list_b=[]
list_c=[]
list_d=[]
def detectAndDisplay(frame):
    # pass the blob through the model and obtain the detections 
    model = cv2.dnn.readNetFromCaffe(prototxt_name, model_name)

    # Resizing to a fixed 300x300 pixels and then normalizing it
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > min_confidence:
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    #print(confidence, startX, startY, endX, endY)
     
                    # draw the bounding box of the face along with the associated
                    # probability
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    face_list.append(endY  - startY)
                    a = (width_20 / face_list[i])
                    #a배만큼 키워야함.
                    need_change.append(a)

                    #가로의 크기
                    Xsize = abs(math.trunc(need_change[i] * people_location[i][1]))
                    #세로의 크기
                    Ysize = abs(math.trunc(need_change[i] * people_location[i][3]))
                    #print(Xsize, Ysize)

                    #사람 부분만 잘라내기
                    dst = img[people_location[i][2]:people_location[i][2] + people_location[i][3], people_location[i][0]:people_location[i][0] + people_location[i][1]].copy()
                    #잘라낸 부분  키우기.
                    dst1 = cv2.resize(dst, (Xsize, Ysize))
                    wid, heig, chann = dst1.shape
                    #print(wid, heig, chann)
                    #4는 인원수 + 1. 이후 지호형 코드 받아서 수정.
                    a = math.trunc(((whitew / 4 * (i+1) - heig)/2) + heig * i)
                    b = math.trunc(((whitew / 4 * (i+1) + heig)/2) + heig * i)
                    c = math.trunc(((whiteh / 4 - Ysize) / 2 + wid / 2))
                    d = math.trunc(((whiteh / 4 + Ysize) / 2 + wid / 2))
                    #왼쪽으로 올라가는거 잡아줌.
                    if(a < 0):
                        b = b - a
                        a = 0

                    #위로 올라가는거 잡아줌.
                    if(c < 0):
                        d = d - c
                        c = 0

                    #반올림 오류 없애줌.
                    while(Xsize != (b-a)):
                        if(Xsize > (b-a)):
                            b = b + 1
                        elif(Xsize < (b-a)):
                            a = a + 1

                    while(Ysize != (d-c)):
                        if(Xsize > (d-c)):
                            d = d + 1
                        elif(Xsize < (d-c)):
                            d = c + 1

                    if(b > whitew):
                        temp = b - whitew
                        b = b - temp
                        a = a - temp
                   #print((b-a, Xsize, (d-c), Ysize))

                    if(d > 685 and b > 1000):
                        d = 685
                        b = 1000
                        dst1 = dst1[c:d, a:b].copy()

                    elif(d > whiteh):
                        d = 685
                        dst1 = dst1[c:d, a:b].copy()
                    elif(b > whitew):
                        b = 1000
                        dst1 = dst1[c:d, a:b].copy()

                    print(a, b, c, d)
                    white[c:d, a:b] = dst1
                    #cv2.imshow("Face size Changed", white)
                   # cv2.rectangle(frame, (startX, startY), (endX, endY),
                   #(0, 255, 0), 2)
                   # cv2.putText(frame, text, (startX, y),
                   #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    
    # show the output image
    #cv2.imshow("Face size Changed", frame)
    cv2.imshow("Face size Changed", white)
    
img = cv2.imread(file_name)


(height, width) = img.shape[:2]


detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()
