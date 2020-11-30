#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 嘴巴检测
import cv2

# 加载
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

if mouth_cascade.empty():
    raise IOError('Unable to load the mouth cascade classifier xml file')

# 读取本地摄像头，这里的参数可以传递
# 1. url，如"rtsp://"
# 2. 本地文件，如"test.mp4"
cap = cv2.VideoCapture(0)

# 进入循环
while True:
    # 读取一帧的视频数据，数据存储在frame中
    ret, frame = cap.read()
    if not ret:
        print('read frame failed.')
        break

    # 灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.9, 3) # 初始 1.7 3
    for (x, y, w, h) in mouth_rects:
        y = int(y - 0.15 * h)
        # 在目标上画框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        break

    # 显示图像
    cv2.imshow('Mouth Detection', frame)

    # c = cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理数据
cap.release()
cv2.destroyAllWindows()