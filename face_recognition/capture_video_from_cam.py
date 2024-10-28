# importing libraries
import cv2
import face_recognition
import PIL.Image
import PIL.ImageDraw

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
        img_name = 'test.jpg'
        cv2.imwrite(img_name, frame)
        image = face_recognition.load_image_file(img_name)
        face_locations = face_recognition.face_locations(image)
        number_of_faces = len(face_locations)
        print("I found {} face(s) in this photograph.".format(number_of_faces))

        for face_location in face_locations:
            pil_image = PIL.Image.fromarray(image)
            # Print the location of each face in this image. Each face is a list of co-ordinates in (top, right, bottom, left) order.
            top, right, bottom, left = face_location
            print(
                "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                      right))

            # Let's draw a box around the face
            draw = PIL.ImageDraw.Draw(pil_image)
            draw.rectangle([left, top, right, bottom], outline="red")

            # Display the image on screen
            # pil_image.show()
            file = open('test2.jpg', 'w')
            pil_image.save(file)
            frame = face_recognition.load_image_file(file)


        cv2.imshow('Frame', frame)
        # if number_of_faces:
        #     break

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