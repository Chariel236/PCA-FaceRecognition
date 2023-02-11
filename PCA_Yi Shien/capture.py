
# # use cam to capture image and save automatically
# import cv2

# cap=cv2.VideoCapture(0)
# path = "Pics"
# falg = 1
# num = 1

# while cap.isOpened():
#     ret_flag,Vshow = cap.read()
#     cv2.imshow("Capture_Test",Vshow)
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('s'):
#        cv2.imwrite("./testing/"+str(num)+".newCapture"+".jpg",Vshow)
#        print("success to save"+str(num)+".jpg")
#        print("-------------------")
#        num += 1
#     elif k == ord(' '):
#         break

# cap.release()

# cv2.destroyAllWindows()

##################

import cv2

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Frame not captured")
        break
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Scale and crop each face in the frame
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        rows, cols = face_gray.shape
        if rows > cols:
            difference = rows - cols
            col_start = 0
            row_start = difference // 2
            col_end = cols
            row_end = rows - difference // 2
        else:
            difference = cols - rows
            col_start = difference // 2
            row_start = 0
            col_end = cols - difference // 2
            row_end = rows
        face_cropped = face_gray[row_start:row_end, col_start:col_end]

    # Display the frame with the faces
    cv2.imshow("Faces", frame)

    # Check if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # If the 'q' key is pressed, save the cropped face and break the loop
        cv2.imwrite("testing.jpg", face_cropped)
        break

# Release the camera and destroy the window
cap.release()
cv2.destroyAllWindows()
