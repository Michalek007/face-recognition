import face_recognition

known_people_files = ['person_1.jpg', 'person_2.jpg', 'person_3.jpg', 'jarek1.jpg']

known_face_encodings = []
for file in known_people_files:
    # Load the known images
    image_of_person = face_recognition.load_image_file(file)

    # Get the face encoding of each person. This can fail if no one is found in the photo.
    person_face_encoding = face_recognition.face_encodings(image_of_person)[0]

    # Create a list of all known face encodings
    known_face_encodings.append(person_face_encoding)


unknown_people_files = ['jarek2.jpg', 'jarek3.jpg', 'unknown_1.jpg', 'unknown_2.jpg', 'unknown_3.jpg', 'unknown_4.jpg', 'unknown_5.jpg', 'unknown_6.jpg', 'unknown_7.jpg', 'unknown_8.jpg']


for file in unknown_people_files:
    # Load the image we want to check
    unknown_image = face_recognition.load_image_file(file)

    # Get face encodings for any people in the picture
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)

    # There might be more than one person in the photo, so we need to loop over each face we found
    for unknown_face_encoding in unknown_face_encodings:

        # Test if this unknown face encoding matches any of the three people we know
        results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.6)

        names = []
        for i in range(len(known_face_encodings)):
            if results[i]:
                names.append(known_people_files[i])

        print(f"Found {names} in the photo: {file}!")
