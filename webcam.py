import cv2
import sys

eyePath = "haarcascade_eye.xml"
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
eye_cascade = cv2.cv2.CascadeClassifier(eyePath)


video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #faces = faceCascade.detectMultiScale(
    #    gray,
    #    scaleFactor=1.1,
    #    minNeighbors=5,
    #    minSize=(30, 30),
    #    flags=cv2.CASCADE_SCALE_IMAGE
    #)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    print(type(faces))
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        print('\a')
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
