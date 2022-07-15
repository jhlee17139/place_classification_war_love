# 사랑과 전쟁 장소 인식

합성곱 신경망을 이용한 장소 인식 코드(학습, 평가, 추론)

사랑과 전쟁 장소 데이터는 드라이브에 업로드

사랑과 전쟁 장소 데이터는 과제(인공지능을 활용한 콘텐츠 창작 기술, 1711152844) 목적으로 구축

## Requirements

pytorch, torchvision, opencv, tqdm 설치


## 실행 준비

폴더를 아래와 같이 정리

드라이브에 업로드된 love_war_place와 실행 코드는 같은 폴더에 위치해야함


## 코드 실행 방법

장소 인식 분류기 학습
- python train.py configs/conv_model


장소 인식 성능 평가
- python eval.py configs/conv_model


장소 인식기로 주석 생성
- python inference.py configs/conv_model
- 주석을 생성하고 싶은 이미지들은 love_war_place/inference/input에 넣으면 됨
- 출력은 love_war_place/inference/output에 생성
- 주석 형식은 pascal voc(XML)으로 생성


## 장소 인식기 성능

- accuracy
- 8가지 장소 인식 성능

| total | car | front_of_buliding | hospital | house | indoor |
| ----- | ------ | ----- | ----------- | ---------- | --------- |
| 81.64 | 57.14  | 70.00 | 55.55      | 89.88     | 71.42      |

| restaurant | rooftop | street |
| ----- | ------ | ----- |
| 73.97 | 44.44  | 91.30 |