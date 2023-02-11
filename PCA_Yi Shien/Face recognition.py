# from sklearn.decomposition import PCA
# import numpy as np
# import cv2
# import os
# import sys

# def read_images(path, sz=None):
#     """Reads images in a given folder, resizes images on the fly if size is given.

#     Args:
#         path: Path to a folder with subfolders representing the subjects (persons).
#         sz: A tuple with the size Resizes 

#     Returns:
#         A list [X, y, folder_names]
#             X: The images, which is a Python list of numpy arrays.
#             y: The corresponding labels (the unique number of the subject, person) in a Python list.
#             folder_names: The names of the folder, so you can display it in a prediction.
#     """
#     c = 0
#     X,y = [], []
#     folder_names = []
#     for dirname, dirnames, filenames in os.walk(path):
#         for subdirname in dirnames:
#             folder_names.append(subdirname)
#             subject_path = os.path.join(dirname, subdirname)
#             for filename in os.listdir(subject_path):
#                 try:
#                     im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
#                     # resize to given size (if given)
#                     if (sz is not None):
#                         im = cv2.resize(im, sz)
#                     X.append(np.asarray(im, dtype=np.uint8))
#                     y.append(c)
#                 except IOError as e:
#                     print("I/O error({0}): {1}".format(e.errno, e.strerror))
#                 except:
#                     print("Unexpected error:", sys.exc_info()[0])
#                     raise
#             c = c+1
#     return [X,y,folder_names]

# def compute_eigenfaces(X):
#     """
#     Compute the mean and eigenfaces from a dataset of faces.

#     Parameters:
#     - X: a list of faces, where each face is a 2D numpy array

#     Returns:
#     - mean: the mean face, as a 2D numpy array
#     - eigenfaces: a list of eigenfaces, where each eigenface is a 2D numpy array
#     """
#     # Convert the list of faces to a 2D numpy array
#     X = np.array(X)

#     # Compute the mean face
#     mean = np.mean(X, axis=0)

#     # Subtract the mean face from each face
#     X = X - mean

#     X = X.reshape(-1, 1)

#     # Perform PCA
#     pca = PCA()
#     pca.fit(X)
#     eigenfaces = pca.components_

#     return mean, eigenfaces

# def recognize_face(X, y, test_image, mean, eigenfaces, threshold=3000.0):
#     """Recognizes a face in an image using PCA.

#     Args:
#         X: A list of numpy arrays representing the images.
#         y: The corresponding labels (the unique number of the subject, person) in a Python list.
#         test_image: The test image to recognize the face in.
#         mean: The mean of the dataset.
#         eigenfaces: A list of eigenfaces computed from the dataset.
#         threshold: Threshold for face recognition.

#     Returns:
#         A tuple (identity, distance), where identity is the predicted identity of the face, and distance is the
#         distance between the face and the closest face in the database (the lower the distance,
#     the closer the match).
#     """
#     # Compute the eigenvector for the test image
#     projection = cv2.face.createEigenFaceRecognizer().computeEigen(test_image, mean)

#     # Initialize minimum distance and identity
#     min_dist = float('inf')
#     identity = None

#     # Loop over all faces in the database
#     for i in range(len(X)):
#         dist = np.linalg.norm(projection - eigenfaces[i])
#         if dist < min_dist:
#             min_dist = dist
#             identity = y[i]

#     # Check if the minimum distance is less than the threshold
#     if min_dist < threshold:
#         return (identity, min_dist)
#     else:
#         return (None, None)

# # Example usage:

# # Load the dataset
# [X, y, folder_names] = read_images("Train")

# # Compute the mean and eigenfaces
# [mean, eigenfaces] = compute_eigenfaces(X)

# # Load the test image
# test_image = cv2.imread("tesing.jpg", cv2.IMREAD_GRAYSCALE)

# # Recognize the face in the test image
# (identity, distance) = recognize_face(X, y, test_image, mean, eigenfaces)

# if identity is not None:
#     print("Identity: {0}".format(folder_names[identity]))
#     print("Distance: {0}".format(distance))
# else:
#     print("Face not recognized.")
##################
import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def show_face(image):
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

def load_images(path, size=(112, 92)):
    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized_img = cv2.resize(img, size)
            images.append(resized_img.flatten())
    return np.array(images)

def compute_eigenfaces(X):
    pca = PCA(n_components=12)
    pca.fit(X)
    eigenfaces = pca.components_
    mean = pca.mean_
    return mean, eigenfaces

def recognize_face(X, mean, eigenfaces, test_image, size=(112, 92)):
    resized_test_image = cv2.resize(test_image, size)
    test_diff = resized_test_image.flatten() - mean
    test_projection = np.dot(test_diff, eigenfaces.T)
    distances = np.linalg.norm(np.dot(X - mean, eigenfaces.T) - test_projection, axis=1)
    min_index = np.argmin(distances)
    return min_index



X = load_images("Train")
test_image = X[0]
show_face(test_image.reshape(112, 92))
[mean, eigenfaces] = compute_eigenfaces(X)
test_image = cv2.imread("testing.jpg", cv2.IMREAD_GRAYSCALE)
resized_test_image = cv2.resize(test_image, (112, 92))
flattened_test_image = resized_test_image.flatten()
predicted_index = recognize_face(X, mean, eigenfaces, flattened_test_image)
labels = os.listdir("Train")
print("Recognized face:", labels[predicted_index])

