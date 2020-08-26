
import cv2
import numpy as np
# import Image
screenLevels = 255.0
def yuv_import(filename,dims,numfrm,startfrm):
    fp=open(filename,'rb')
    blk_size = int(np.prod(dims) *3/2)
    #fp.seek(blk_size*startfrm,0)
    Y=[]
    U=[]
    V=[]
    print(dims[0])
    print(dims[1])
    d00=dims[0]//2
    d01=dims[1]//2
    print(d00)
    print(d01)
    Yt=np.zeros((dims[0],dims[1]),np.uint8,'C')
    Ut=np.zeros((d00,d01),np.uint8,'C')
    Vt=np.zeros((d00,d01),np.uint8,'C')
    # yuv=zeros((dims[0]*3//2,dims[1]))
    for i in range(numfrm):
        for m in range(dims[0]):
            for n in range(dims[1]):
                #print m,n
                num=fp.read(1)
                Yt[m,n]=ord(num)
        for m in range(d00):
            for n in range(d01):
                Ut[m,n]=ord(fp.read(1))
        for m in range(d00):
            for n in range(d01):
                Vt[m,n]=ord(fp.read(1))
        Y=Y+[Yt]
        U=U+[Ut]
        V=V+[Vt]
    fp.close()
    return (Y,U,V)
if __name__ == '__main__':
    width=480
    height=272
    data=yuv_import('E:\\project\\2020_Project\\partiledetect\\YUV\\cuc_view_480x272.yuv',(height,width),1,0)
    rgb=np.zeros((height,width,3))
    YY=data[0][0]
    cv2.imshow("show",YY)

    b=np.fromfile('E:\\project\\2020_Project\\partiledetect\\YUV\\cuc_view_480x272.yuv',np.uint8,height*width*3//2)
    c=b.reshape((height*3//2,width))
    rgb=cv2.cvtColor(c,cv2.COLOR_YUV2BGR_I420,rgb)
    cv2.imshow("rgb",rgb)
    cv2.imwrite("rgb.jpg",rgb)
    cv2.waitKey(0)