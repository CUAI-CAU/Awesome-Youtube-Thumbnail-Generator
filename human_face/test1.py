import cv2
import face_recognition
import pickle
import time
import os
import cv2
import numpy as np
import math
import numpy as np
file_name = 'video/test_only4.mp4'
encoding_file = 'encodings.pickle'

# Either cnn  or hog. The CNN method is more accurate but slower. HOG is faster but less accurate.
model_method = 'hog'
output_name = 'video/output_' + model_method + '.avi'
people_count = 0
face_count = 0
total_face_size = 0

cap = cv2.VideoCapture(file_name)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
total_size = width * height
fra = 0

#모든 프레임, 얼굴 수, 사람 수, 크기, 평균 크기가 저장되는 리스트
fra_list= []
face_list = []
people_list = []
size_list = []
averageSize_list = []


#recommand_people = face_list = people_list 일때 저장되는 리스트들
rec_fra_list = []
rec_face_list = []
rec_people_list = []
rec_size_list = []
rec_averageSize_list = []

#사람 수, 얼굴 수, 
path = 'output'
def face_detection(image):
    start_time = time.time()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    boxes = face_recognition.face_locations(rgb,
        model=model_method)
    encodings = face_recognition.face_encodings(rgb, boxes)

    # initialize the list of names for each face detected

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
            encoding)

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face

            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
        
        # update the list of names

    # loop over the recognized faces
    for (top, right, bottom, left) in (boxes):
        global face_count
        global total_face_size
        global total_size
        global fra
        # draw the predicted face name on the image
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        face_count += 1
        face_size = (right-left) * (bottom - top)
        total_face_size += face_size
        print("left, top, right, bottom", left, top, right, bottom)
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    print("people count", face_count)
    if(face_count == 0):
        fra_list.append(fra)
        face_list.append(face_count)
        size_list.append(total_face_size / total_size * 100)
        averageSize_list.append(0)
    else:
        #print("average of face", total_face_size / people_count)
        #print("화면 전체 크기 대비 얼굴 비율", total_face_size / total_size * 100)
        #print("화면 전체 크기 대비 얼굴 비율(개인)", total_face_size / total_size / people_count * 100)
        fra_list.append(fra)
        face_list.append(face_count)
        size_list.append(total_face_size / total_size * 100)
        averageSize_list.append(total_face_size / total_size / face_count * 100)

    #프레임, 사람 수, 얼굴 총 크기, 얼굴 평균 크기가 담긴 리스트.

    fra += 1
    face_count = 0
    total_face_size = 0
    # show the output image
    image = cv2.resize(image, None, fx=0.5, fy=0.5)
    cv2.imshow("Recognition", image)
    
    # if the video writer is None *AND* we are supposed to write
    # the output video to disk initialize the writer
    global writer
    if writer is None and output_name is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_name, fourcc, 24,
                (image.shape[1], image.shape[0]), True)

    # if the writer is not None, write the frame with recognized
    # faces to disk
    if writer is not None:
        writer.write(image)

def human_detection(frame):
    start_time = time.time()
    img = cv2.resize(frame, None, fx=0.4, fy=0.4)
    total_h, total_w, c = img.shape
    height, width, channels = img.shape
    cv2.imshow("Original Image", img)

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            min_confidence = 0.5
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

    #노이즈 제거(공통된 박스 제거)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        global people_count
        global fra
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}".format(classes[class_ids[i]], confidences[i]*100)
            print(i, label)
            color = colors[i]
            #이후 주석은 사람인 주식을 표시해주는 주석.
            
            if (classes[class_ids[i]] == 'person'):
                people_count = people_count + 1
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
                
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds, ".format(process_time))
    fra += 1
    print("count, frame", people_count, fra)
    cv2.imshow("YOLO Video", img)
    people_list.append(people_count)
    people_count = 0



# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
#classes.append("people")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# load the known faces and embeddings
data = pickle.loads(open(encoding_file, "rb").read())

#-- 2. Read the video stream
cap = cv2.VideoCapture(file_name)
writer = None
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    face_detection(frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap = cv2.VideoCapture(file_name)
fra = 0

while True:
    ret, frame = cap.read()
    if frame is None:
        for i in range(fra):
            #이후 수정되어야 하는 변수.
            recommand_people = 3
            if(face_list[i] == recommand_people and people_list[i] == recommand_people):
                rec_fra_list.append(fra_list[i])
                rec_face_list.append(face_list[i])
                rec_people_list.append(people_list[i])
                rec_size_list.append(size_list[i])
                rec_averageSize_list.append(averageSize_list[i])
                    
        max_face = max(rec_averageSize_list)
        recom_index = rec_averageSize_list.index(max_face)
        
        print("We recommand", rec_fra_list[recom_index], "frame!")
        cap.set(cv2.CAP_PROP_POS_MSEC, fra_list[recom_index] / 30 * 1000)
        success,image = cap.read()
        cv2.imwrite(os.path.join(path , 'recommand_picture.png'), image)
            
        #print('--(!) No captured frame -- Break!')
        # close the video file pointers
        cap.release()
        # close the writer point
        writer.release()
        break
    
    human_detection(frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#여기서부터 사람 크기 조정 시작.
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
width_20 = w * 0.07
height_20 = h * 0.07
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

                    Xsize = abs(math.trunc(need_change[i] * people_location[i][1]))
                    Ysize = abs(math.trunc(need_change[i] * people_location[i][3]))
                    print(Xsize, Ysize)
                    dst = img[people_location[i][2]:people_location[i][2] + people_location[i][3], people_location[i][0]:people_location[i][0] + people_location[i][1]].copy()
                    dst1 = cv2.resize(dst, (Xsize, Ysize), interpolation=cv2.INTER_NEAREST)
                    wid, heig, chann = dst1.shape
                    print(wid, heig, chann)
                    #3은 인원수 + 1.
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
                    print((b-a, Xsize, (d-c), Ysize))
                    print(a, b, c, d)
                    
                    #print(a, b, c, d)
                    white[c:d, a:b] = dst1
                    cv2.imshow("Face size Changed", white)
                   # cv2.rectangle(frame, (startX, startY), (endX, endY),
                        #    (0, 255, 0), 2)
                   # cv2.putText(frame, text, (startX, y),
                   #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    
    # show the output image
    #cv2.imshow("Face size Changed", frame)
 
img = cv2.imread(file_name)


(height, width) = img.shape[:2]


detectAndDisplay(img)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.destroyAllWindows()
