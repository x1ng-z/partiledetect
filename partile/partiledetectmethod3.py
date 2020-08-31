import cv2 as cv
import numpy as np
import socket
import numpy as np
import cv2 as cv
import ctypes
import struct
import time
import imagepost

DEBUG = False


def read(size):
    ret = memoryview(bytearray(size))
    remain = size  # 30
    while True:
        data = client.recv(remain)
        length = len(data)  # 10
        ret[size - remain: size - remain + length] = data  # [30-30:30-30+10]
        if len(data) == remain:
            break
        remain -= len(data)
    return ret


if __name__ == '__main__':
    nextsendtime = time.time()
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("192.168.156.26", 8079))
    data = [0x00, 0x00, 0x00, 0x01,
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x01, ord('p'), ord('u'), ord('l'), ord('l'),
            0x00]
    bytedata = bytearray(data)
    client.send(bytedata)
    # picturesize = client.rev(4)
    oldtime = time.time()
    while True:
        time.sleep(0.1)
        newtime = time.time()
        if newtime - oldtime > 1:
            start = time.time()
            # print("shold get pic")
            oldtime = newtime
            picturesize = read(4)  # 4字节的图片大小
            sum = 0
            a = picturesize[3]
            sum += (picturesize[3] & 0xffffffff) << 24
            b = picturesize[2]
            sum += (picturesize[2] & 0xffffffff) << 16
            c = picturesize[1]
            sum += (picturesize[1] & 0xffffffff) << 8
            d = picturesize[0]
            sum += (picturesize[0] & 0xffffffff)
            # pclength=(picturesize[3]&0xffffffff)<<24+(picturesize[2]&0xffffffff)<<16+(picturesize[1]&0xffffffff)<<8+(picturesize[0]&0xffffffff)
            picture = read(sum)
            image = np.asarray(picture, dtype="uint8")
            image = cv.imdecode(image, cv.IMREAD_COLOR)
            if DEBUG:
                cv.imshow("success", image)
            row = 216
            col = 236
            startrow = 620
            startcol = 620
            src = image[startrow:startrow + row, startcol:startcol + col]  # 1080 1920
            # cv.imshow('src', src)
            # keyboard = cv.waitKey(0)
            # if keyboard == 'q' or keyboard == 27:
            #     break

            # src=cv.imread("E:\\project\\2020_Project\\1595634455(1).jpg")
            cv.imshow("origin", src)

            gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)

            # cv.imshow("gray",gray)
            # laplas=np.zeros_like(gray)
            # laplas=cv.Laplacian(gray,ddepth=cv.CV_32F,dst=laplas,ksize=3)
            # cv.imshow("laplas",laplas)

            cannyedeg = cv.Canny(gray, 150, 150 * 2, apertureSize=3, L2gradient=True)
            if DEBUG:
                cv.imshow("cannyedeg", cannyedeg)

            edge = gray.copy()
            edge[cannyedeg == 255] = 0
            if DEBUG:
                cv.imshow("cannyedegresult", edge)

            # binaryImg=np.zeros_like(edge)
            # cv.threshold(edge, 120,255,cv.THRESH_BINARY,binaryImg)
            # cv.imshow("binary image", binaryImg)

            # kernel = np.ones((3, 3), np.uint8)
            # opening = binaryImg#cv.morphologyEx(binaryImg, cv.MORPH_OPEN, kernel, iterations=1)
            # cv.imshow("opening image", opening)

            dist_transform = np.zeros_like(edge)
            dist_transform = cv.distanceTransform(edge, cv.DIST_L1, cv.DIST_MASK_3)
            normdis = np.zeros_like(edge)
            normdis = cv.normalize(dist_transform, dist_transform, 0, 1, cv.NORM_MINMAX);
            ret, sure_fg = cv.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            if DEBUG:
                cv.imshow("sure_fg", sure_fg * 20)

            contours, hierarchy = cv.findContours(sure_fg, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
            makers = np.zeros((src.shape[0], src.shape[1]), np.int32)
            print(len(contours))
            for i, contour in enumerate(contours):
                area = cv.contourArea(contour)
                # print("index={},Area={}".format(i,area))
                cv.drawContours(makers, contours, int(i), (i + 1, i + 1, i + 1), cv.FILLED)

            # cv.circle(makers, (5, 5), 3, (255, 255, 255), -1)
            # cv.imshow("my markers", np.uint8(makers))

            markers = cv.watershed(src, makers)

            colors = np.random.randint(0, 255, (len(contours), 3))

            dst = np.zeros_like(src)

            for row in range(markers.shape[0]):
                for col in range(markers.shape[1]):
                    index = markers[row, col]
                    if ((index > 0) and (index <= len(contours))):
                        dst[row, col] = colors[index - 1, :]
                    else:
                        src[row, col] = [0, 255, 0]
                        dst[row, col] = [0, 0, 0]
                    pass

            # cv.imshow("Final Result1", src)
            # cv.imshow("Final Result2", dst)
            icontours, hierarchy = cv.findContours(makers, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
            drawcolor = np.zeros_like(src)
            colors = np.random.randint(0, 255, (len(icontours), 3))

            stoneNum = 0
            stoneArea = 0
            TotalArea = 0
            pickstone = np.zeros_like(src)
            print(len(icontours))
            for i, icontour in enumerate(icontours):
                area = cv.contourArea(icontour)
                TotalArea = area + TotalArea
                # print("index2={},Area2={}".format(i, area))
                aa = (colors[i, 0], colors[i, 1], colors[i, 2])
                cv.drawContours(drawcolor, icontours, int(i), (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])),
                                cv.FILLED)
                if area < 600:
                    stoneNum = stoneNum + 1
                    stoneArea = stoneArea + area
                    cv.drawContours(pickstone, icontours, int(i),
                                    (int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])), cv.FILLED)

            if DEBUG:
                cv.imshow("Final Result3", np.uint8(drawcolor))

            if DEBUG:
                cv.putText(pickstone, str(stoneNum) + "/" + str(stoneArea) + "/" + str(TotalArea),
                           (int(src.shape[0] / 4), int(src.shape[1] / 4)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255))
                cv.putText(pickstone, str((stoneArea / TotalArea) * 100),
                           (int(src.shape[0] / 4), int(src.shape[1] * 3 / 4)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (255, 255, 255))
                print("total=" + str(TotalArea) + " stonearea=" + str(stoneArea))
            cv.imshow("Final Result4", np.uint8(pickstone))
            if nextsendtime <= time.time():
                nextsendtime = time.time() + 60 * 5
                imagepost.image_post("http://127.0.0.1:8080/iovedio/analysispic/1", src, np.uint8(pickstone))
            imagepost.analyzeresult_post("http://127.0.0.1:8080/iovedio/analysisdata/1",
                                         {"stoneNum": stoneNum, "stoneArea": stoneArea})

            # cv.waitKey(0)

            # foxpointM = foxpointM + 1
            # # print('*****MaterialArea******', area)
            # if area > maxMeterialArea:
            #     maxMeterialArea = area
            #     maxMeterialArea_M = []
            #     maxMeterialArea_M = contour

            # ret, makers1 = cv.connectedComponents(sure_fg)
            # cv.imshow("mask1",np.uint8(makers1))

            # markers = makers1 + 1

            # Now mark the region of unknow with zero;
            # unknow = cv.subtract(sure_bg, sure_fg)
            # markers[unknow == 255] = 0
            # markers3 = cv2.watershed(img1, markers)

            # cv.waitKey(1)
            keyboard = cv.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break
            print("cost time=" + str(time.time() - start))

    # capture = cv.VideoCapture("E:\\project\\2020_Project\\视频分析\\生料磨分析\\vlc-record-2020-07-23-15h22m43s-rtsp___192.168.200.160_Streaming_Channels_1-.mp4")
    # if not capture.isOpened:
    #     print('Unable to open: ')
    #     exit(0)
    #
    # while True:
    #     ret, frame = capture.read()
    #     if frame is None:
    #         break
    #     cv.imshow('Frame', frame)
    #     keyboard = cv.waitKey(30)
    #     if keyboard == 'q' or keyboard == 27:
    #         break

    pass
