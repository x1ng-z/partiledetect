import cv2 as cv
import numpy as np




if __name__ == '__main__':

    capture = cv.VideoCapture("E:\\project\\2020_Project\\视频分析\\生料磨分析\\vlc-record-2020-07-23-15h22m43s-rtsp___192.168.200.160_Streaming_Channels_1-.mp4")
    if not capture.isOpened:
        print('Unable to open: ')
        exit(0)

    while True:
        ret, frame = capture.read()#216,236
        if frame is None:
            break
        cv.imshow('Frame', frame)
        row=216
        col=236
        startrow=620
        startcol=620
        src=frame[startrow:startrow+row,startcol:startcol+col]#1080 1920
        # cv.imshow('src', src)
        # keyboard = cv.waitKey(0)
        # if keyboard == 'q' or keyboard == 27:
        #     break

    # src=cv.imread("E:\\project\\2020_Project\\1595634455(1).jpg")
        cv.imshow("origin", src)

        gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)

        cv.imshow("gray",gray)
        laplas=np.zeros_like(gray)
        laplas=cv.Laplacian(gray,ddepth=cv.CV_32F,dst=laplas,ksize=3)
        cv.imshow("laplas",laplas)

        cannyedeg=cv.Canny(gray,80,80*3,apertureSize=3,L2gradient=True)
        cv.imshow("cannyedeg", cannyedeg)

        edge=gray - cannyedeg
        cv.imshow("cannyedegresult", edge)

        binaryImg=np.zeros_like(edge)
        cv.threshold(edge, 180,255,cv.THRESH_BINARY,binaryImg)
        cv.imshow("binary image", binaryImg)

        kernel = np.ones((3, 3), np.uint8)
        opening = binaryImg#cv.morphologyEx(binaryImg, cv.MORPH_OPEN, kernel, iterations=1)
        cv.imshow("opening image", opening)

        dist_transform=np.zeros_like(opening)
        dist_transform=cv.distanceTransform(opening,cv.DIST_L1,cv.DIST_MASK_3)
        normdis=np.zeros_like(opening)
        normdis=cv.normalize(dist_transform, dist_transform, 0, 1, cv.NORM_MINMAX);
        ret, sure_fg = cv.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        cv.imshow("sure_fg",sure_fg*20)

        contours, hierarchy=cv.findContours(sure_fg,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        makers=np.zeros((src.shape[0],src.shape[1]),np.int32)
        for i, contour in enumerate(contours):
            area = cv.contourArea(contour)
            print("index1={},Area1={}".format(i,area))
            cv.drawContours(makers,contours,int(i),(i+1,i+1,i+1),cv.FILLED)

        # cv.circle(makers, (5, 5), 3, (255, 255, 255), -1)
        cv.imshow("my markers", np.uint8(makers))

        markers=cv.watershed(src,makers)

        colors=np.random.randint(0,255,(len(contours),3))

        dst = np.zeros_like(src)

        for row in range(markers.shape[0]):
            for col in range(markers.shape[1]):
                index =markers[row,col]
                if((index>0) and (index<=len(contours))):
                    dst[row,col]=colors[index-1,:]
                else:
                    src[row, col]=[0,255,0]
                    dst[row, col]=[0,0,0]
                pass


        cv.imshow("Final Result1", src)
        cv.imshow("Final Result2", dst)


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

        cv.waitKey(1)





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