from keras.models import load_model
import cv2
import numpy as np
from yoloface import face_analysis

# Load model
model_age = load_model('./model/model_age.hdf5')

a = []

def detect_video(url):
    frame = cv2.VideoCapture(url)
    
    while True:
        _, img = frame.read()
        img = cv2.flip(img, 1)
        face=face_analysis()
        _,box,_=face.face_detection(frame_arr=img,frame_status=True,model='tiny')
        for x,y,w,h in box:
            cv2.rectangle(img, (x,y), (x+h,y+w), (0,255,0), 2)
            img_detect = cv2.resize(img[y:y+w, x:x+h], dsize=(50, 50)).reshape(1, 50, 50, 3)

            #Detect Age
            age = np.round(model_age.predict(img_detect/255.))[0][0]
            a.append(int(age))

            #Draw
            cv2.putText(img, f'Age: {age}', (x-5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (np.random.randint(150, 230),np.random.randint(50, 150),np.random.randint(80, 180)), 1, cv2.LINE_AA)
        cv2.imshow('detect', img)
        if cv2.waitKey(1) == ord('q'):
            break
        if len(a)==5:
            break
    print(a)
    print(max(a))
    print(np.mean(a))
    b = np.mean(a)
    frame.release()
    cv2.destroyAllWindows()

detect_video(0)
