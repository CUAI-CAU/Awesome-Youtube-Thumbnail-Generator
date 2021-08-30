# Awesome-Youtube-Thumbnail-Generator

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FCUAI-CAU%2FAwesome-Youtube-Thumbnail-Generator&count_bg=%2383C4E9&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Hits%21&edge_flat=true)](https://hits.seeyoufarm.com)


[Paper]() | [Project]() | [Presentation]() 


**This repository contains source codes which used for "Final Project for CUAI 4th Summer Conference :surfing_man:".**

Paper and Presentation are in Korean.  

### Our Team 
 - **Nahyuk Lee**[@](mailto:nahyuk0113@gmail.com) (School of Computer Science & Engineering :desktop_computer:, Chung-Ang Univ.)
 - **Hyejin Jang**[@](mailto:wkdgpwls617@gmail.com) (School of Integrative Engineering 	:test_tube:, Chung-Ang Univ.)
 - **Wonyoung Choi**[@](mailto:) (School of Computer Science & Engineering :desktop_computer:, Chung-Ang Univ.)
 - **Jiho Hong**[@](mailto:hgjiho@naver.com) (School of Mechanical Engineering 	:hammer_and_wrench:, Chung-Ang Univ.)

## Application

![ex1](docs/result2.jpg)
Original youtube clip is in [here](https://www.youtube.com/watch?v=kQJ1pnVIwss&ab_channel=KBSKpop)!


## Pipeline
![pipeline](docs/pipeline.png)

## Code

### Install dependencies
We recommend you to use Anaconda that already including mandotory packages.
```
conda create -n youthumb python=3.8
conda activate youthumb
python -m pip install -r requirements.txt
```
Our code was tested with Python 3.8, Pytorch 1.9.0, CUDA 11.1

### Generating Thumbnails
Before you generating thumbnails, save your video under "input", and run the command. Generated thumbnail will be stored in "output" directory.
```
python main.py <your_video_name.mp4>
```

