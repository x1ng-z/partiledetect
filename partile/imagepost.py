import requests
import cv2 as cv
import os
# 客户端发送
def image_post(url,originImage,resultImage):
    # data = {"type_code": type_code,
    #         "area_id": area_id}
    # 以文件的格式上传，节省传输时间
    # multiple_files = [
    #     ('images', ('foo.png', open('foo.png', 'rb'), 'image/png')),
    #     ('images', ('bar.png', open('bar.png', 'rb'), 'image/png'))]
    multiple_files = [
        ('images', ("file_name.jpg", cv.imencode(".jpg", originImage)[1].tobytes(), "image/jpg")),
        ('images', ("file_name.jpg", cv.imencode(".jpg", resultImage)[1].tobytes(), "image/jpg"))]
    # res = requests.post(url=url, files=multiple_files, data=data)
    res = requests.post(url=url, files=multiple_files)


rgb1=cv.imread("E:\\project\\2020_Project\\partiledetect\\YUV\\rgb.jpg")
rgb2=cv.imread("E:\\project\\2020_Project\\partiledetect\\YUV\\rgb.jpg")
image_post("http://127.0.0.1:8080/iovedio/analysispic/1",rgb1,rgb2)