import os
import pickle
import re
import time

import PIL
import face_recognition


##idea from https://github.com/ageitgey/face_recognition/issues/243

def current_sec_time():
    return round(time.time() * 1000)


known_faces_filenames = []
known_face_names = []
known_face_encodings = {}

print("########################################")
start_time = current_sec_time();
for dirpath, dirnames, filenames in os.walk('assets/'):
    known_faces_filenames.extend(filenames)

for filename in known_faces_filenames:
    print(filename)
    if filename == "desktop.ini":
        continue
    face = face_recognition.load_image_file('assets/' + filename)
    try:
        face_encoding = face_recognition.face_encodings(face)[0]
    except IndexError:
        print("Error " + filename)
    except PIL.UnidentifiedImageError:
        print("Error " + filename)
    known_face_encodings[re.sub("[0-9]", '', filename[:-4])] = face_encoding
end_time = current_sec_time()

print("init time " + str((end_time - start_time) / 1000) + "s")
print("########################################")
with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(known_face_encodings, f)
print("File created")
print("########################################")