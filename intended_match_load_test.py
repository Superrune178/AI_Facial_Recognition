# libraries imported
import os
import re
import cv2
import face_recognition
from encoding import read_known_user, get_image_encodings
from cam_scan import facial_detection, match
import numpy as np
import matplotlib.pyplot as plt

known_users_path = "appdata/imgs/users/"
input_path = "appdata/imgs/known-users/"
known_users = []
num_tests = 5
test_iter_correct = []
test_iter_incorrect = []

# train the faces
print("Training and encoding faces...")
read_known_user(known_users)
known_enc = get_image_encodings(known_users_path)
print("Finished encoding")

for i in range(num_tests):

    print("Beginning test %s" % (i + 1))

    # make counters for correct and incorrect recognitions
    correct = 0
    incorrect = 0

    # iterate through all of the images in test directory
    for img in os.listdir(input_path):

        # load the input image
        print("Loading image...")
        input_img = face_recognition.load_image_file(input_path + img)
        print("Image loaded")

        print("Processing image...")
        # find face in frame and get encodings
        face_locations, shown_enc = facial_detection(input_img)

        print("Comparing faces...")
        # match the faces to known faces
        info = match(shown_enc[0], known_enc, known_users)

        # get the name of the person in photo from file name
        name_in_file = img[:-6]

        if name_in_file == info[0]:
            # increment correct counter for correct match
            correct += 1
        else:
            # increment incorrect counter for not matching properly
            incorrect += 1
            print("Intended name: " + name_in_file)
            print("File name: " + img)
            print("Given name: %s" % info[0])

        print("Comparing done")
    
    # add number of correct and incorrect matches to test_iter arrays
    test_iter_correct.append(correct)
    test_iter_incorrect.append(incorrect)

print("Making bar chart...")
# create plot
fig, ax = plt.subplots()
index = np.arange(num_tests)
bar_width = 0.35
opacity = 0.8

# create correct and incorrect rectangles
rects1 = plt.bar(index, test_iter_correct, bar_width,
alpha=opacity,
color='g',
label='Correct matches')

rects2 = plt.bar(index + bar_width, test_iter_incorrect, bar_width,
alpha=opacity,
color='r',
label='Incorrect matches')

# labels
plt.xlabel('Test iteration')
plt.ylabel('Number of matches')
plt.title('Correct vs Incorrect matches of existing users using load method')
plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))
plt.legend()

# make plot
plt.tight_layout()
plt.savefig('testdata/intended_matches_load_test.jpg', dpi=400)
plt.show()