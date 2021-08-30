import torchvision
from PIL import Image
from torchvision import transforms as T
import torch
import random
import numpy as np
import cv2
import sys
import pandas as pd
import math
from video2frame import video2frame
import os

mask_list = []
box_list = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(img_path, threshold=0.5, url=False):

    img = Image.open(img_path) # This is for local images

    transform = T.Compose([T.ToTensor()]) # Turn the image into a torch.tensor
    img = transform(img)
    img = img.to(device) # Only if GPU, otherwise comment this line
    pred = model([img]) # Send the image to the model. This runs on CPU, so its going to take time

    #Let's change it to GPU
    #pred = pred.cpu() # We will just send predictions back to CPU
    # Now we need to extract the bounding boxes and masks
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    
    return masks, pred_boxes, pred_class

def random_color_masks(image):
    # I will copy a list of colors here
    colors = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180], [250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image==1], g[image==1], b[image==1] = colors[random.randrange(0, 10)]
    colored_mask = np.stack([r,g,b], axis=2)
    return colored_mask

def instance_segmentation(img_path, threshold=0.5, rect_th=3,
                          text_size=3, text_th=3, url=False):
    global mask_list, box_list
    masks, boxes, pred_cls = get_prediction(img_path, threshold=threshold, url=url)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For working with RGB images instead of BGR

    for i in range(len(masks)):
        if pred_cls[i] == 'person':
            rgb_mask = random_color_masks(masks[i])
            img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            mask_list.append(masks[i])
            box_list.append(boxes[i])
    return img, pred_cls, masks[i]

def recommend(file_path):
    global mask_list, box_list, device, model, COCO_INSTANCE_CATEGORY_NAMES

    video2frame(file_path)

    #frame number 바꿔가면서 오류 발생하는지 아닌지 판별
    mask_list = []
    box_list = []

    file_path = os.path.splitext(file_path)[0]+'/'
    file_list = os.listdir(file_path)
    #print(file_list)

    number_of_people = [0 for i in range(len(file_list))]

    error_count = 0
    for i in range(len(file_list)):
        mask_list = []
        try:
            img, pred_classes, masks = instance_segmentation(file_path+file_list[i], rect_th=5, text_th=4)
        except:
            print('Skip frame')
            error_count += 1
        number_of_people[i] = len(mask_list)
        print('[Human Detection] %d of humans in %d frame'%(len(mask_list), i))

    col = ['number_of_people']
    df = pd.DataFrame(number_of_people, columns=col)


    # 4분위수 기준 지정하기     
    q25, q75 = np.quantile(df['number_of_people'], 0.25), np.quantile(df['number_of_people'], 0.75)          

    iqr = q75 - q25    
    cut_off = iqr * 1.5          

    # lower와 upper bound 값 구하기     
    lower, upper = q25 - cut_off, q75 + cut_off

    # 1사 분위와 4사 분위에 속해있는 데이터 각각 저장하기
    data1 = df[df['number_of_people'] > upper]     
    data2 = df[df['number_of_people'] < lower]    

    # 이상치 총 개수 구하기
    df_outliers_upper = df[df['number_of_people']>upper]
    df_outliers_lower = df[df['number_of_people']<lower]

    index_outliers = df_outliers_upper.index.tolist() + df_outliers_lower.index.tolist()

    sum = 0

    for i in range(len(file_list)):
        if i in index_outliers:
            continue  
        
        sum = sum + number_of_people[i]
    recommend_people = math.ceil(sum / len(file_list))

    #print(recommend_people)

    return recommend_people
