import socket
import numpy as np
import cv2 as cv
import ctypes
import struct
import time

# data=[0x00,0x00,0x00,0x01,
#           0x00,0x00, 0x00,0x00,
#           0x00,0x01, ord('p'),ord('u'),ord('l'),ord('l'),
#           0x00]
# # a = bytearray(data)
# a = bytearray('111',encoding='ascii')
# ma = memoryview(a)
# imag=np.asarray(ma,dtype="uint8")



#data = input("请输入给服务器发送的数据")

'''
lengthFieldOffset   = 0 (= the length of HDR1)
lengthFieldLength   = 4
lengthAdjustment    = 10 (= the length of HDR1 and HDR2)
initialBytesToStrip = 4 (= the length of HDR1 + LEN)
BEFORE DECODE (n bytes)                       AFTER DECODE (n-4 bytes)
+------------+--------------------+------------------+----------------+      +------------------+----------------+
| Length     |  HDR1 height,width |HDR2 camid method | Actual Content |----->| HDR1 height,width| Actual Content |
| 4 bytes    |   2+2 bytes        |2+4 bytes         |    n bytes     |      |  2+2 bytes       |   b bytes      |
    '''


# ctypes.cre
# bytedata=bytearray(data)
# aa=bytes.fromhex(data)
# b = ""
# for i in range(len(data)):
#     b += chr(data[i])
# struct.pack
# client.send(bytedata)

# picture=client.rev(1024)

    # data.encode('hex')
    # image = np.asarray(bytearray(res.content), dtype="uint8")
    # image = cv.imdecode(image, cv2.IMREAD_COLOR)


    # client.send(data.tobytes());
    # info = client.recv(1024)
    # print("服务器说：", info)


def read(size):
    ret = memoryview(bytearray(size))
    remain = size#30
    while True:
        data = client.recv(remain)
        length = len(data)#10
        ret[size - remain: size - remain + length] = data#[30-30:30-30+10]
        if len(data) == remain:
            break
        remain -= len(data)
    return ret


if __name__ == '__main__':
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
        try:
            time.sleep(0.1)
            newtime = time.time()
            if newtime-oldtime>1:
                print("shold get pic")
                oldtime=newtime
                picturesize=read(4)
                sum=0
                a=picturesize[3]
                sum+=(picturesize[3]&0xffffffff)<<24
                b = picturesize[2]
                sum += (picturesize[2] & 0xffffffff) << 16
                c = picturesize[1]
                sum += (picturesize[1] & 0xffffffff) <<8
                d = picturesize[0]
                sum += (picturesize[0] & 0xffffffff)
                # pclength=(picturesize[3]&0xffffffff)<<24+(picturesize[2]&0xffffffff)<<16+(picturesize[1]&0xffffffff)<<8+(picturesize[0]&0xffffffff)
                picture=read(sum)
                image = np.asarray(picture, dtype="uint8")
                image = cv.imdecode(image, cv.IMREAD_COLOR)
                cv.imshow("success",image)

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

                cv.imshow("gray", gray)
                #laplas = np.zeros_like(gray)
                #laplas = cv.Laplacian(gray, ddepth=cv.CV_32F, dst=laplas, ksize=3)
                #cv.imshow("laplas", laplas)

                cannyedeg = cv.Canny(gray, 50, 50 * 3, apertureSize=3, L2gradient=True)
                cv.imshow("cannyedeg", cannyedeg)
                #cv.waitKey(0)
                edge = gray.copy()
                edge[cannyedeg != 255]=0
                cv.imshow("cannyedegresult", edge)

                # ret, strangestone = cv.threshold(gray, 0.5 * gray.max(), 255, cv.THRESH_BINARY)
                # cv.imshow("strangestone", strangestone)
                # edge[strangestone==255]=255
                # cv.imshow("strangestone and edg", edge)
                # cv.waitKey(0)


                #burlpic=cv.bilateralFilter(edge,5,20,20)
                #cv.imshow("bilateralFilter", burlpic)
                #cv.waitKey(0)

                #burlpic#np.zeros_like(burlpic)

                # kernel = np.ones((1, 1), np.uint8)
                # opening = cv.morphologyEx(edge, cv.MORPH_OPEN, kernel, iterations=1)
                # cv.imshow("opening image", opening)
                #cv.waitKey(0)

                i,binaryImg=cv.threshold(cannyedeg, 0.4*cannyedeg.max(), 255, cv.THRESH_BINARY)
                cv.imshow("binary image", binaryImg)
                #cv.waitKey(0)
                # kernel = np.ones((3, 3), np.uint8)
                # opening = binaryImg  # cv.morphologyEx(binaryImg, cv.MORPH_OPEN, kernel, iterations=1)
                # cv.imshow("opening image", opening)
                #cv.waitKey(0)
                dist_transform = np.zeros_like(binaryImg)
                dist_transform = cv.distanceTransform(binaryImg, cv.DIST_L2, cv.DIST_MASK_3)

                normdis = np.zeros_like(dist_transform)
                normdis = cv.normalize(dist_transform, dist_transform, 0, 1, cv.NORM_MINMAX)
                ret, sure_fg = cv.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
                sure_fg = np.uint8(sure_fg)
                cv.imshow("sure_fg", sure_fg * 20)
                #cv.waitKey(0)
                contours, hierarchy = cv.findContours(sure_fg, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
                makers = np.zeros((src.shape[0], src.shape[1]), np.int32)
                for i, contour in enumerate(contours):
                    area = cv.contourArea(contour)
                    print("index={},Area={}".format(i, area))
                    cv.drawContours(makers, contours, int(i), (i + 1, i + 1, i + 1), cv.FILLED)

                # cv.circle(makers, (5, 5), 3, (255, 255, 255), -1)
                cv.imshow("my markers", np.uint8(makers))

                markers = cv.watershed(src, makers)

                colors = np.random.randint(0, 255, (len(contours), 3))

                dst = np.zeros_like(src)

                # contours, hierarchy=cv.findContours(makers,cv.RETR_FLOODFILL,cv.CHAIN_APPROX_NONE)
                # cv.waitKey(0)



                for row in range(markers.shape[0]):
                    for col in range(markers.shape[1]):
                        index = markers[row, col]
                        if ((index > 0) and (index <= len(contours))):
                            dst[row, col] = colors[index - 1, :]
                        else:
                            src[row, col] = [0, 255, 0]
                            dst[row, col] = [0, 0, 0]
                        pass

                cv.imshow("Final Result1", src)
                cv.imshow("Final Result2", dst)
                cv.waitKey(0)


                cv.findContours()

                k = cv.waitKey(30) & 0xff
                if (k == 27) and False:
                    break
        except Exception:
            print(Exception)