import cv2
import numpy as np

modelFile = "E:\\Better Face Auth with OpenCV\\itsMaFace\\main\\res10_300x300_ssd_iter_140000.caffemodel"
configFile = "E:\\Better Face Auth with OpenCV\\itsMaFace\\main\\deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX 
while(True):
    ret, img = cap.read()
    if ret == True:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        height, width = img.shape[:2]
        
        # detect faces in the image
        
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)),
                                        1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces1 = net.forward()
        
        # display faces on the original image
        for i in range(faces1.shape[2]):
            confidence = faces1[0, 0, i, 2]
            if confidence > 0.5:
                box = faces1[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
        

        cv2.imshow("DNN", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
           
cap.release()
cv2.destroyAllWindows()