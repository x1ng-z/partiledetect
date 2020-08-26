import cv2
import time
# cap = cv2.VideoCapture("D:\\360MoveData\\Users\\zaixz\\Desktop\\石灰石破碎口2_1560470653_FEF44932\\石灰石破碎口2_1867C048_1560470332_5.mp4")#"rtsp://admin:jdhs12345@192.168.130.141//Streaming/Channels/1"
# "摄像头：192.168.200.111  admin  hk123456"
#rtsp://admin:admin12345@192.168.200.160//Streaming/Channels/1
# "rtsp://10.10.10.3:554/pag://10.10.10.3:7302:e0cd654fe95a41c9be420b9b454bd1eb:0:SUB:TCP?streamform=rtp"
# cap = cv2.VideoCapture("rtsp://10.10.10.3:554/pag://10.10.10.3:7302:f87a584ce2c54879893403b8715f6af3:0:SUB:TCP?streamform=rtp")
cap = cv2.VideoCapture("rtsp://admin:jdhs12345@192.168.130.141//Streaming/Channels/1")#("rtsp://admin:admin12345@192.168.200.160//Streaming/Channels/1")

oldtime = time.time()

print (cap.isOpened())
#rtsp://10.10.10.2:554/hikvision://10.10.13.14:8000:0:0?cnid=4&pnid=4&username=1&password=1
i=352
while cap.isOpened():
    ret,frame = cap.read()
    newtime = time.time()
    cv2.namedWindow("frame")
    cv2.imshow("frame",frame)
    if newtime - oldtime > 1:
        i=i+1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_cut_car = gray_frame[0:1080, 0:600]
        #cv2.imwrite('picture'+str(i)+'.png', gray_cut_car)
        oldtime=newtime
    cv2.waitKey(1)
