from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import numpy as np

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load pre-trained FaceNet model
resnet = InceptionResnetV1(pretrained='casia-webface').eval()

known_face_images = ('jarek1.jpg', 'morawiecki1.jpg')
known_face_embeddings = []
for image_file in known_face_images:
    # Load an image containing faces
    img = Image.open(image_file)

    # Detect faces in the image
    boxes, _ = mtcnn.detect(img)

    # If faces are detected, extract embeddings
    if boxes is not None:
        aligned = mtcnn(img)
        embeddings = resnet(aligned).detach()
        known_face_embeddings.append(embeddings[0])

unknown_face_images = ('jarek2.jpg', 'jarek3.jpg', 'morawiecki2.jpg')
for image_file in unknown_face_images:
    # Load an image containing faces
    img = Image.open(image_file)

    # Detect faces in the image
    boxes, _ = mtcnn.detect(img)

    # If faces are detected, extract embeddings
    if boxes is not None:
        aligned = mtcnn(img)
        embeddings = resnet(aligned).detach()
        for embedding in embeddings:
            i = 0
            for known_embedding in known_face_embeddings:
                # distance = np.linalg.norm(embedding-known_embedding)
                distance = (embedding - known_embedding).norm().item()
                if distance < 1.0:
                    print(f'Found {known_face_images[i]} in {image_file}')
                i += 1
