
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the model
model = load_model('model.h5')

# Define mediapipe Face detector

face_detection = mp.solutions.face_detection.FaceDetection(0.5)

# Detection function



def get_detection(frame):

    height, width, channel = frame.shape

    # Convert frame BGR to RGB colorspace

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect results from the frame
    
    result = face_detection.process(imgRGB)


    try:
        for count, detection in enumerate(result.detections):

            # print(detection)

        
            # Extract bounding box information 
            score = detection.score
            box = detection.location_data.relative_bounding_box
            score = str(round(score[0]*100, 2))
            x, y, w, h = int(box.xmin*width), int(box.ymin * height), int(box.width*width), int(box.height*height)
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            #cv2.rectangle(frame, (x, y), (x+w, y-25), (0, 0, 255), -1)

            #cv2.putText(frame, score, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        count += 1
        print("Found ",count, "Faces!")
            
    # If detection is not available then pass 
    except:
        pass

    return x, y, w, h,count,result

CATEGORIES = ['no_mask', 'mask']
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    
    try:
        x, y, w, h,count,result = get_detection(frame)
        for detection in enumerate(result.detections):
            img = detection.copy()
        
            crop_img = img[y:y+h, x:x+w]
        
            crop_img = cv2.resize(crop_img, (100, 100))
        
            crop_img = np.expand_dims(crop_img, axis=0)
        
            # get the prediction from the model.
            prediction = model.predict(crop_img)
            print(prediction)
            index = np.argmax(prediction)
            res = CATEGORIES[index]
            if index == 0:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, res, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.8, color, 2, cv2.LINE_AA)
        

    except:
        pass
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()



