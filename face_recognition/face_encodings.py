import face_recognition

images = ('people.jpg', 'img.png', 'img_1.png', 'img_2.png', 'img_3.png')

for image_name in images:
    # Load the jpg files into numpy arrays
    image = face_recognition.load_image_file(image_name)

    # Generate the face encodings
    face_encodings = face_recognition.face_encodings(image)

    if len(face_encodings) == 0:
        # No faces found in the image.
        print("No faces were found.")

    else:
        # Grab the first face encoding
        first_face_encoding = face_encodings[0]

        # Print the results
        print(first_face_encoding)
