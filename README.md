# Face compare
使用```face_recognition```建構的人臉辨識系統
# 安裝
- 下載 [dlib](https://pypi.org/simple/dlib/)
```
pip install ./dlib-19.8.1-cp36-cp36m-win_amd64.whl
pip install face-recognition

pip install pyarrow
pip install fastparquet

pip install opencv-python
```
# 使用
做辨識需先準備人臉的資料, 人臉的編碼需要一段時間, 所以把程式分成兩個部分:
- ```encode_face.py``` 負責將照片中的人臉編碼,  並將結果儲存.
- ```main.py``` 讀取人臉編碼, 並實時辨識人臉並標註名稱.

## encode_face.py

- 將照片依照下面的格式儲存, 請確保照片中只有一個人.
    - ./face/name/example1.jpg
    - ./face/name/example2.jpg
    - ...
- 使用```encode_face.py```編碼資料夾內的人臉, 並輸出```face_encoding_data```供主程式使用.

```
python ./encode_face.py
```
## main.py
使用OpenCV擷取鏡頭畫面, 實時比對```face_encoding_data```內的資料並標註, 提供的照片越多, 則會更多次的做比對, 減少誤判.
```
python ./main.py
```