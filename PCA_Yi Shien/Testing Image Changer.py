import cv2
import os
import numpy as np

def detect_face(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained classifier for detecting faces
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If no face is detected, return the original image
    if len(faces) == 0:
        return None, None

    # Take the face with the largest area
    face = max(faces, key=lambda x: x[2] * x[3])

    # Extract the face from the image
    x, y, w, h = face
    face = img[y:y+h, x:x+w]

    # Resize the face to a fixed size
    face = cv2.resize(face, (200, 200))

    # Return the face and the face ROI
    return face, (x, y, w, h)

def preprocess_folder(src_folder, dest_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Loop through all the images in the source folder
    for filename in os.listdir(src_folder):
        # Load the image
        img = cv2.imread(os.path.join(src_folder, filename))

        # Detect and preprocess the face
        face, roi = detect_face(img)
        if face is None:
            print("No face detected in image: ", filename)
            continue

        # Center the face in the image
        x, y, w, h = roi
        x = x + int((w - face.shape[1]) / 2)
        y = y + int((h - face.shape[0]) / 2)

        # Crop the image to only contain the face
        face = img[y:y+240, x:x+240]

        # Save the preprocessed image
        dest_path = os.path.join(dest_folder, filename)
        cv2.imwrite(dest_path, face)

# Example usage
preprocess_folder('Pics', 'Train')
