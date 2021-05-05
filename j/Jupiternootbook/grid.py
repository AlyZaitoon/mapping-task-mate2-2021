import numpy as np
import cv2 as cv
def Hist(img):
    H=np.zeros(shape=(256,1))
    s=img.shape
    for i in range(s[0]):
        for j in range(s[1]):
            k=img[i,j]
            H[k,0]=H[k,0]+1
    return H
def stretchHistogram(channel):
    histogram=Hist(channel)
    x=histogram.reshape(1,256)
    y=np.zeros((1,256))
    for i in range(256):
        if x[0,i]==0:
            y[0,i]=0
        else:
            y[0,i]=i
    min=np.min(y[np.nonzero(y)])
    max = np.max(y[np.nonzero(y)])
    stretch=np.round(((255-0)/(max-min))*(y-min))
    stretch[stretch<0]=0
    stretch[stretch>255]=255
    s=image.shape
    for i in range(s[0]):
        for j in range(s[1]):
            k=channel[i,j]
            channel[i,j]=stretch[0,k]
    channel=cv.convertScaleAbs(channel,None,1.2,30)
    return channel
def drawongrid(row,column,shape,color):
   point=((4*row+2)*33,(4*column+2)*33)
   if(shape=='circle') and locations[row][column]==0:
       cv.circle(img,point,40,color,2)
       locations[row][column]=1
   if(shape=='sqaure') and locations[row][column]==0:
       cv.rectangle(img,(point[0]-40,point[1]-40),(point[0]+40,point[1]+40),color,2)
       locations[row][column] = 1
   if(shape=='rectangle') and locations[row][column]==0:
       cv.rectangle(img,(point[0]-40,point[1]-20),(point[0]+40,point[1]+20),color,2)
       locations[row][column] = 1
   if(shape=='ellipse') and locations[row][column]==0:
       cv.ellipse(img,(point[0]+66,point[1]),(100,50),0,0,360,color,2)
       locations[row][column] = 1
       locations[row+1][column] = 1

def empty(a):
    pass
#Creating the window for the trackbars and the trackbars for each parameter in HSV
cv.namedWindow("HSV")
cv.resizeWindow("HSV",640,240)
cv.createTrackbar("HUE Min","HSV",0,179,empty)
cv.createTrackbar("HUE Max","HSV",179,179,empty)
cv.createTrackbar("SAT Min","HSV",0,255,empty)
cv.createTrackbar("SAT Max","HSV",255,255,empty)
cv.createTrackbar("VALUE Min","HSV",0,255,empty)
cv.createTrackbar("VALUE Max","HSV",255,255,empty)


if __name__ == '__main__':
    #grid
    img=np.zeros([4*99,4*297,3])
    # Grid lines at these intervals (in pixels)
    # dx and dy can be different
    dx, dy = 4 * 33, 4 * 33

    # Custom (rgb) grid color
    grid_color = [255, 0, 0]

    # Modify the image to include the grid
    img[:, ::dy, :] = grid_color
    img[::dx, :, :] = grid_color
    images=[cv.imread("frametest290a.jpg"),cv.imread("frametest290b.jpg"),cv.imread("frametest417a.jpg"),cv.imread("frametest498a.jpg")]
    locations=np.array([[0 ,0 ,0],
                       [0, 0, 0],
                       [0, 0, 0]])
    for i,image in enumerate(images):
        #enhancement
        (r, g, b) = cv.split(image)
        r = stretchHistogram(r)
        g = stretchHistogram(g)
        b = stretchHistogram(b)
        image = cv.merge([r, g, b])

        #trackbars to test values
        h_redmin = cv.getTrackbarPos("HUE Min", "HSV")
        h_redmax = cv.getTrackbarPos("HUE Max", "HSV")
        s_redmin = cv.getTrackbarPos("SAT Min", "HSV")
        s_redmax = cv.getTrackbarPos("SAT Max", "HSV")
        v_redmin = cv.getTrackbarPos("VALUE Min", "HSV")
        v_redmax = cv.getTrackbarPos("VALUE Max", "HSV")
        imgHsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
   #lower values for pink and red masks
        lowerred = np.array([0, 0, 160])
        upperred = np.array([155, 255, 255])
        lowerpink=np.array([0,0,0])
        upperpink=np.array([102,255,255])
   #getting values within these masks
        maskpink = cv.inRange(imgHsv, lowerpink, upperpink)
        maskred = cv.inRange(imgHsv, lowerred, upperred)
    #anding these values to get values that are not red or pink
        resultpink = cv.bitwise_and(image, image, mask=maskpink)
        resultred = cv.bitwise_and(image, image, mask=maskred)
    #making image - these values to get red and pink values
        fimgpink = image - resultpink
        fimgred = image - resultred
    #showing them in a stack
        hStack = np.hstack([image, fimgred,fimgpink])
        cv.imshow('Hstack', hStack)
    #converting images to grayscale and getting contours
        imgbgr=cv.cvtColor(fimgred,cv.COLOR_HSV2BGR)
        imggrayred=cv.cvtColor(imgbgr,cv.COLOR_BGR2GRAY)
        imgbgr = cv.cvtColor(fimgpink, cv.COLOR_HSV2BGR)
        imggraypink = cv.cvtColor(imgbgr, cv.COLOR_BGR2GRAY)



        contoursr, _ = cv.findContours(imggrayred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contoursp, _ = cv.findContours(imggraypink, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for c in contoursr:
            # compute the bounding box of the contour and then draw the coressponding shape on the grid
            if 1000<cv.contourArea(c)<2000:
                (x, y, w, h) = cv.boundingRect(c)

                cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                height , width, channels=image.shape
                centre=(x+(x+w))/2

                if centre < width/2-50 :
                    column=0
                elif centre > width/2+50 :
                    column=2
                elif width/2-50 <centre< width/2+50:
                    column=1
                drawongrid(i, column, "circle", [255, 0, 0])

        for c in contoursp:
            # compute the bounding box of the contour and then draw the coressponding shape on the grid
            print(cv.contourArea(c))
            if cv.contourArea(c)>300:
                (x, y, w, h) = cv.boundingRect(c)
                print("ok")
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                height , width, channels=image.shape
                centre=(x+(x+w))/2
                if centre < width/2-50 :
                    column=0
                elif centre > width/2+50 :
                    column=2
                elif width / 2 - 50 < centre < width / 2 + 50:
                    column=1
                drawongrid(i,column,"ellipse",[0,0,255])
                print(i,column)
# Show the result

    cv.imshow("ab",image)
    cv.imshow('a',img)
    cv.waitKey(0)
