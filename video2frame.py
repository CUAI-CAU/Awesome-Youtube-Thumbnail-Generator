import cv2
import os
import shutil
import sys

def video2frame(video_path):
    new_path = os.path.split(video_path)[0]+'/'+os.path.splitext(os.path.split(video_path)[1])[0]
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.mkdir(new_path)
    vidcap = cv2.VideoCapture(video_path)

    count = 0
    while(vidcap.isOpened()): 
        ret, image = vidcap.read() 
        if image is None:
            break
        # 30프레임당 하나씩 이미지 추출
        if(int(vidcap.get(1)) % 30 == 0): 
            image = cv2.resize(image, (960, 540)) # 이미지 사이즈 960x540으로 변경 
            print(new_path+'/'+"frame%d.png" % count)
            cv2.imwrite(new_path+'/'+"frame%d.png" % count, image) 
            count += 1

    vidcap.release()

# Command 내 인자 개수 확인
if len(sys.argv) != 2:
    print("Insufficient arguments")
    sys.exit()

# input File 열기
file_path = sys.argv[1]

video2frame(file_path)