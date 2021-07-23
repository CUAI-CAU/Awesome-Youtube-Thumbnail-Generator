import cv2
import face_recognition
import pickle
import time
import os
file_name = 'video/test.mp4'
encoding_file = 'encodings.pickle'

# Either cnn  or hog. The CNN method is more accurate but slower. HOG is faster but less accurate.
model_method = 'hog'
output_name = 'video/output_' + model_method + '.avi'
people_count = 0
total_face_size = 0

cap = cv2.VideoCapture(file_name)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
total_size = width * height
fra = 0

fra_list= []
people_list = []
size_list = []
averageSize_list = []
path = 'output'
def detectAndDisplay(image):
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
        global people_count
        global total_face_size
        global total_size
        global fra
        # draw the predicted face name on the image
        y = top - 15 if top - 15 > 15 else top + 15
        color = (0, 255, 0)
        line = 2
        cv2.rectangle(image, (left, top), (right, bottom), color, line)
        y = top - 15 if top - 15 > 15 else top + 15
        people_count += 1
        face_size = (right-left) * (bottom - top)
        total_face_size += face_size
        print("left, top, right, bottom", left, top, right, bottom)
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    print("people count", people_count)
    if(people_count == 0):
        print("this frame has no people")
    else:
        print("average of face", total_face_size / people_count)
        print("화면 전체 크기 대비 얼굴 비율", total_face_size / total_size * 100)
        print("화면 전체 크기 대비 얼굴 비율(개인)", total_face_size / total_size / people_count * 100)
        fra_list.append(fra)
        people_list.append(people_count)
        size_list.append(total_face_size / total_size * 100)
        averageSize_list.append(total_face_size / total_size / people_count * 100)

    #프레임, 사람 수, 얼굴 총 크기, 얼굴 평균 크기가 담긴 리스트.

    fra += 1
    people_count = 0
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
        max_face = max(averageSize_list)
        recom_index = averageSize_list.index(max_face)
        
        print("We recommand", fra_list[recom_index], "frame!")
        cap.set(cv2.CAP_PROP_POS_MSEC, fra_list[recom_index] / 30 * 1000)
        success,image = cap.read()
        cv2.imwrite(os.path.join(path , 'recommand_picture.png'), image)
            
        #print('--(!) No captured frame -- Break!')
        # close the video file pointers
        cap.release()
        # close the writer point
        writer.release()
        break
    detectAndDisplay(frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

