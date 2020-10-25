import cv2
import numpy as np
import face_recognition
import glob
import os
import pandas as pd

CHECK_RANGE = 5
CHECK_POINT = int((CHECK_RANGE - 1) / 2)

temp_img = []
temp_location = []
temp_enconding = []
temp_compare = []

names = os.listdir('./face')

know_face_encoding_table = pd.DataFrame(columns=range(CHECK_RANGE))
for name in names:
    encoding_list = {}
    
    image_list = glob.glob(os.path.join("./face/" + name + "/", "*.jpg"))
    for k in range(len(image_list)):
        image = face_recognition.load_image_file(image_list[k])
        encoding = face_recognition.face_encodings(image)[0]
        encoding_list[k] = encoding
        
    know_face_encoding_table = know_face_encoding_table.append(pd.Series(encoding_list, name=name))
    
know_face_encoding_table.columns = know_face_encoding_table.columns.astype(str)
know_face_encoding_table.to_parquet("./face_encoding_data")

print(know_face_encoding_table)
# know_face_encoding_table.to_csv("face_encoding.csv")