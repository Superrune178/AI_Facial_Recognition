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
input_path = "appdata/imgs/unknown-users/"
known_users = []
num_tests = 5
test_iter_matched = []
test_iter_unmatched = []

# train the faces
print("Training and encoding faces...")
read_known_user(known_users)
known_enc = get_image_encodings(known_users_path)
print("Finished encoding")

for i in range(num_tests):
    unknown_matched, unknown_unmatched = 0, 0

    for img in os.listdir(input_path):
        print("Loading Images...")

        input_img = face_recognition.load_image_file(input_path + img)
        print("Image Loaded")

        print("Processing Image")
        face_locations, shown_enc = facial_detection(input_img)

        print("Comparing Faces...")
        info = match(shown_enc[0], known_enc, known_users)

        if info[0] == 'Unknown':
            unknown_unmatched += 1
        else:
            unknown_matched += 1
            print("File name: " + img)

    test_iter_unmatched.append(unknown_unmatched)
    test_iter_matched.append(unknown_matched)

print("Making bar chart")

fig, ax = plt.subplots()
index = np.arange(num_tests)
bar_width = 0.35
opacity = 0.8

rect1 = plt.bar(index, test_iter_unmatched, bar_width, \
        alpha=opacity, color='g', label='Unmatched')

rect2 = plt.bar(index + bar_width, test_iter_matched, bar_width, \
        alpha=opacity, color='r', label='Matched')

plt.xlabel('Test Iterations')
plt.ylabel('Number of matches')
plt.title('Correct vs. Incorrect Matches of Unknown Users')
plt.xticks(index + bar_width, ('1', '2', '3', '4', '5'))
plt.legend()

#plt.tight_layout()
plt.savefig('testdata/unknown_matches.jpg', dpi=400)
plt.show()
