"""
Face Swap AI in Python with InsightFace
https://www.youtube.com/watch?v=a8vFMaH2aDw
"""

import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

# get versions
insightface.__version__  # 0.7.3
np.__version__           # 1.24.3

## 1. Detect faces
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# Get sample image
img = ins_get_image('t1')

# Display image
plt.imshow(img[:,:,::-1])
plt.show()

# Detect faces
faces = app.get(img)
# - returns a list of dicts, one for each face

len(faces) # 6 faces
faces[0].keys()


# Crop and plot faces
fig, axs = plt.subplots(1, 6, figsize=(12, 5))
for i, face in enumerate(faces):
    bbox = face['bbox']
    bbox = [int(b) for b in bbox]
    axs[i].imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
    axs[i].axis('off')
plt.show()


# 2. Face swapping
swapper = insightface.model_zoo.get_model(
    'inswapper_128', download=False, download_zip=False
)
# - Use reddit link to download model. 
#   Save it in ~/.insightface/models/<model_name>/<model_name>.onnx

# Example: Choosing the first face, apply over all others
source_face = faces[0]

# Print face
bbox = source_face['bbox']
bbox = [int(b) for b in bbox]
plt.imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2], ::-1])
plt.show()

# Swap faces
res = img.copy()
for face in faces:
    res = swapper.get(res, face, source_face, paste_back=True)

# Plot swapped faces
plt.imshow(res[:, :, ::-1])
plt.show()

# Crop and plot faces
res = []
for face in faces:
    _img, _ = swapper.get(img, face, source_face, paste_back=False)
    res.append(_img)
res = np.concatenate(res, axis=1)
fig, ax = plt.subplots(figsize=(15, 5))
ax.imshow(res[:, :, ::-1])
ax.axis('off')
plt.show()

# ----------------------------------------------------------
## 3. Using my face
FILEPATH = '/Users/andres/Library/CloudStorage/Dropbox/Files/Media/Photos/SocialMedia/Profiles/Andres-2022.JPG'
andres = cv2.imread(FILEPATH)

plt.imshow(andres[:, :, ::-1])
plt.show()

# Detect the face
andres_faces = app.get(andres)
andres_face = andres_faces[0]

# Replace faces in friends image
res = img.copy()
for face in faces:
    res = swapper.get(res, face, andres_face, paste_back=True)
fig, ax = plt.subplots()
ax.imshow(res[:, :, ::-1])
ax.axis('off')
plt.show()


# 4. Swapping to another photo
FILEPATH = '/Users/andres/Library/CloudStorage/Dropbox/Files/Media/Photos/Private/Babes/2016-03-25 00.00.00, [E], Jennifer Leibovici.jpg'
soho = cv2.imread(FILEPATH)
plt.imshow(soho[:, :, ::-1])
plt.show()

faces = app.get(soho)
res = soho.copy()
for face in faces:
    res = swapper.get(res, face, andres_face, paste_back=True)
fig, ax = plt.subplots()
ax.imshow(res[:, :, ::-1])
plt.show()


# 5. Functions

def swap_n_show(img1_fn, img2_fn, app, swapper, 
                plot_before=True, plot_after=True):
    """
    Uses face swapper to swap faces in two different images.

    plot_before: if True shows the images before the swap
    plot_after: if True shows the images after the swap

    returns images with swapped faces.

    Assumes one face per image.
    """
    img1 = cv2.imread(img1_fn)
    img2 = cv2.imread(img2_fn)

    if plot_before:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1[:, :, ::-1])
        axs[0].axis('off')
        axs[1].imshow(img2[:, :, ::-1])
        axs[1].axis('off')
        plt.show()
    
    # Do the swap
    face1 = app.get(img1)[0]
    face2 = app.get(img2)[0]

    img1_ = img1.copy()
    img2_ = img2.copy()
    if plot_after:
        img1_ = swapper.get(img1_, face1, face2, paste_back=True)
        img2_ = swapper.get(img2_, face2, face1, paste_back=True)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img1_[:, :, ::-1])
        axs[0].axis('off')
        axs[1].imshow(img2_[:, :, ::-1])
        axs[1].axis('off')
        plt.show()
    
    return img1_, img2_



# path_sc = '/Users/andres/Library/CloudStorage/Dropbox/Files/Media/Photos/Private/Babes/2015-05-28 01.57.58, [C], 000_iOS.jpg'
# path_ara = '/Users/andres/Library/CloudStorage/Dropbox/Files/Media/Photos/SocialMedia/Profiles/Andres-2022.JPG'
# _ = swap_n_show(path_ara, path_sc, app, swapper)


def swap_n_show_same_img(img1_fn, 
                        app, swapper, 
                        plot_before=True, plot_after=True):
    """
    Uses face swapper to swap faces in the same image.

    plot_before: if True shows the images before the swap
    plot_after: if True shows the images after the swap

    returns images with swapped faces.

    Assumes one face per image.
    """
    img1 = cv2.imread(img1_fn)

    if plot_before:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(img1[:, :, ::-1])
        ax.axis('off')
        plt.show()

    # Do the swap
    faces = app.get(img1)
    face1 = faces[0]
    face2 = faces[1]

    img1_ = img1.copy()
    if plot_after:
        img1_ = swapper.get(img1_, face2, face1, paste_back=True)
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(img1_[:, :, ::-1])
        ax.axis('off')
        plt.show()

    return img1_


# PHOTO = '/Users/andres/Library/CloudStorage/Dropbox/Files/Media/Pictures/Digital Library/Photos/2007-04-13 17.22.27.JPG'
# _ = swap_n_show_same_img(PHOTO, app, swapper)





