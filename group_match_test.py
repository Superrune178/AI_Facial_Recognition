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
input_path = "appdata/imgs/groups/"
known_users = []
num_tests = 5
test_iter_all = []
test_iter_not_all = []
num_faces = 25

# train the faces
print("Training and encoding faces...")
read_known_user(known_users)
known_enc = get_image_encodings(known_users_path)
print("Finished encoding")

for i in range(num_tests):

    print("Beginning test %s" % (i + 1))

    # make counters for when all faces are matched and not
    all_m = 0
    not_all_m = 0

    # iterate through all of the images in test directory
    for img in os.listdir(input_path):

        # load the input image
        print("Loading image...")
        input_img = face_recognition.load_image_file(input_path + img)
        print("Image loaded")

        # find face in frame and get encodings
        print("Processing image...")
        face_locations, shown_enc = facial_detection(input_img)
        num_face_in_frame = len(shown_enc)

        # check if number of faces matched is equal to intended number
        if num_faces == num_face_in_frame:
            # increment all_m counter
            all_m += 1
        else:
            # increment not_all_m counter
            not_all_m += 1
            print("Given number of faces = %s" % num_face_in_frame)
            print("File name: " + img)
        
        print("Comparing done")
    
    # add number of all and not all matched to test_iter arrays
    test_iter_all.append(all_m)
    test_iter_not_all.append(not_all_m)

print("Making bar chart...")
# create plot
fig, ax = plt.subplots()
index = np.arange(num_tests)
bar_width = 0.35
opacity = 0.8

# create all and not all matched rectangles
rects1 = plt.bar(index, test_iter_all, bar_width,
alpha=opacity,
color='g',
label='All faces matched')

rects2 = plt.bar(index + bar_width, test_iter_not_all, bar_width,
alpha=opacity,
color='r',
label='Missing some faces')

# labels
plt.xlabel('Test iteration')
plt.ylabel('Number of complete matches')
plt.title('Successfulness of group scanning')
plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))
plt.legend()

# make plot
plt.tight_layout()
plt.savefig('testdata/group_match_test.jpg', dpi=400)
plt.show()