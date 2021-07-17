# Awesome-Youtube-Thumbnail-Generator
Repository for mobility team B's CUAI summer conference arxiv


### [Schedule]
- 썸네일 배경 이미지 추천 시스템 **(Due : ~7/17)**
  - **Target** : 유퀴즈 Youtube Clip ([Youtube Link](https://www.youtube.com/watch?v=FlmIK9KNb9g&ab_channel=tvNDENT))
  - **Keywords** : Human Segmentation, Masking etc.
  - **Packages** : PyTorch, Pre-trained Segmentation Model, OpenCV etc.

### Requirements

#### Create Virtual Environments & Install Library
```
$ conda create -n cuai python=3.8
$ conda activate cuai
$ pip install -r requirements.txt
```
---

### TO DO
- Frame 추천 시스템
  - Frame 내 사람 수를 계산하여 평균의 올림? ~~~ (Outlier 제거) ... 지호
  - 얼굴 면적 크기를 계산하여 영상 크기에 대한 사람의 비율 고려한 프레임 추천 ... 혜진/원용
- STT(Speech to Text) 
  - 대본 만들기 ... 나혁
  - 대본에서 핵심 키워드 추출하여 단어 재조합하여 썸네일 자막 만들기

다음 회의 날짜 : 7월 23일(금) 오후 10시 ZOOM ~ 코로나 조심 ㅠㅅ
