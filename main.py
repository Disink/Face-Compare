import face_recognition
import cv2
import threading
import time
import pandas as pd

source_image_list = []
compared_image_list = []


class FaceCompareThread(threading.Thread):
    def __init__(self):
        super(FaceCompareThread, self).__init__()
        self.status = True

    def load_data(self):
        self.know_face_encoding_table = pd.read_parquet("./face_encoding_data") 
        self.know_face_encoding_table.columns = self.know_face_encoding_table.columns.astype(int)

        self.names = self.know_face_encoding_table.index.values.tolist()
        self.know_face_encodings = self.know_face_encoding_table.loc[:, 0].tolist()

    def run(self):
        while True:
            if len(source_image_list) > 0:
                source_image = source_image_list[0]
                source_image_list.pop(0)

                # Source face lcoations
                source_face_location_list = face_recognition.face_locations(source_image)

                # Source face encoding
                source_face_encoded_list = face_recognition.face_encodings(source_image)

                # Soucre face compare know face
                compare_list = []
                for source_face_encoded in source_face_encoded_list:
                    results = face_recognition.compare_faces(self.know_face_encodings, source_face_encoded)
                    for j in range(len(results)):
                        if results[j] == True:
                            name = self.names[j]
                            check_face_encodings = self.know_face_encoding_table.loc[name, :].tolist()
                            check_face_encodings = list(filter((None).__ne__, check_face_encodings))
                            results = face_recognition.compare_faces(check_face_encodings, source_face_encoded)
                            Correct_rate = results.count(True) / len(results)
                            compare_list.append("{} {}".format(name, str(Correct_rate)))

                    if True not in results:
                        compare_list.append('Unknown')

                source_compare_list = compare_list

                image = source_image
                for j in range(len(source_face_encoded_list)):
                    top, right, bottom, left = source_face_location_list[j]
                    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                    name = source_compare_list[j]
                    cv2.putText(image, name, (left-10, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                compared_image_list.append(image)

            if self.status == False:
                break


if __name__=="__main__":
    cap = cv2.VideoCapture(0)

    face_compare = FaceCompareThread()
    face_compare.load_data()
    face_compare.start()

    while True:
        ret, camera_image = cap.read()

        source_image_list.append(camera_image)    

        if len(compared_image_list) > 0:
            cv2.imshow('Face Recognition', compared_image_list[0])
            compared_image_list.pop(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            face_compare.status = False
            cv2.destroyAllWindows()
            cap.release()
            break

        time.sleep(0.5)