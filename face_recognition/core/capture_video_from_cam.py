# importing libraries
import cv2
import face_recognition
import PIL.Image
import PIL.ImageDraw


known_people_files = ['ja1.png']
known_face_encodings = [[-0.07922871,  0.07347926,  0.07330955, -0.07746898, -0.10256002,  0.09254359
, -0.00420567, -0.02823596,  0.18466702, -0.12624657,  0.215611,  0.04081507
, -0.22378498,  0.05657957,  0.02125797,  0.16693091, -0.12130598, -0.14183116
, -0.12952705, -0.04955628,  0.00744192,  0.15316783, -0.08369063,  0.02199927
, -0.11231472, -0.34465063, -0.04898923, -0.06567798,  0.04523248, -0.00719175
,  0.01418718,  0.07091115, -0.11058504,  0.05261102,  0.03533869,  0.02386813
, -0.12284546, -0.02050791,  0.25580022,  0.06384338, -0.2319127,   0.05436496
,  0.03260221,  0.27184924,  0.12524927,  0.01012565,  0.069898,  0.0034213
,  0.04771238, -0.32783988,  0.03546181,  0.16170207,  0.1161802,   0.09885479
,  0.06843481, -0.17647666, -0.03748315,  0.0793236 , -0.18109347,  0.13364547
,  0.04432567, -0.13467628, -0.03243503, -0.08036192,  0.20359749,  0.09745923
, -0.12083552, -0.17668936,  0.17198794, -0.17706965, -0.14245737,  0.08657922
, -0.13593404, -0.17123754, -0.29542106,  0.03535696,  0.40706006,  0.16372582
, -0.17718048,  0.01269848,  0.00969725, -0.0364431,   0.04928671,  0.07223933
, -0.05444835, -0.06178099, -0.0431998,   0.02515533,  0.28222537,  0.00600456
, -0.06785253,  0.2181395,   0.05751051,  0.04363113, -0.02344766,  0.04308319
,  0.01543958, -0.10048939, -0.10826267,  0.02907036, -0.04649808, -0.04668545
, -0.00740039,  0.14863998, -0.19065502,  0.29534224, -0.04012441, -0.0216911
, -0.04374131,  0.12953761, -0.1159091,   0.03239874,  0.20410243, -0.18947734
,  0.19919057,  0.20620306,  0.07283804,  0.09093265,  0.0543308,   0.00402228
, -0.03449102,  0.00941846, -0.26827535, -0.15897664,  0.03538355, -0.08036391
,  0.05192119,  0.10031028]
]
# for file in known_people_files:
#     # Load the known images
#     image_of_person = face_recognition.load_image_file(file)
#
#     # Get the face encoding of each person. This can fail if no one is found in the photo.
#     person_face_encoding = face_recognition.face_encodings(image_of_person)[0]
#
#     # Create a list of all known face encodings
#     known_face_encodings.append(person_face_encoding)


# unknown_people_files = ['ja2.jpg', 'ja3.jpg']
# for file in unknown_people_files:
#     # Load the image we want to check
#     unknown_image = face_recognition.load_image_file(file)
#
#     # Get face encodings for any people in the picture
#     unknown_face_encodings = face_recognition.face_encodings(unknown_image)
#
#     # There might be more than one person in the photo, so we need to loop over each face we found
#     for unknown_face_encoding in unknown_face_encodings:
#
#         # Test if this unknown face encoding matches any of the three people we know
#         results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.6)
#
#         names = []
#         for i in range(len(known_face_encodings)):
#             if results[i]:
#                 names.append(known_people_files[i])
#
#         print(f"Found {names} in the photo: {file}!")





# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video file")

# Read until video is completed
while (cap.isOpened()):

    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        frame_file = 'frame.jpg'
        cv2.imwrite(frame_file, frame)

        unknown_image = face_recognition.load_image_file(frame_file)
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
            if names:
                print(f"Found {names} in the frame!")

                face_locations = face_recognition.face_locations(frame)
                for face_location in face_locations:
                    top, right, bottom, left = face_location

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with the name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, str(names), (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)

        cv2.imshow('Frame', frame)
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()