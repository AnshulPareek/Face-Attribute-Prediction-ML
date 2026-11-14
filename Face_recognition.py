import cv2 as cv
import joblib
import numpy as np
import os
from skimage.feature import hog
import pandas as pd

f1 = "D:\\Anshul\\Anshul Codes\\Projects\\Semester Projects\\Face_recognition\\Models\\model_age.pkl"
f2 = "D:\\Anshul\\Anshul Codes\\Projects\\Semester Projects\\Face_recognition\\Models\\model_gd.pkl"
f3 = "D:\\Anshul\\Anshul Codes\\Projects\\Semester Projects\\Face_recognition\\Models\\model_race.pkl"
model_age = joblib.load(f1)
model_gd = joblib.load(f2)
model_race = joblib.load(f3)

def RealTime():
    camera = cv.VideoCapture(0)
    while True:
        ret,frame = camera.read()
        cv.imshow("Face_Recognition",frame)
        if ret:
            img = cv.resize(frame, (200, 200))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
            features = features.reshape(1,-1)
            dummy_columns = [f'f{i}' for i in range(features.shape[1])]
            features_df = pd.DataFrame(features, columns=dummy_columns)
        if cv.waitKey(1) == ord('q'):
            break
    camera.release()
    cv.destroyAllWindows()
    return features_df
def random_photo():
    folder_path = "D:\\Anshul\\Anshul Codes\\Projects\\Semester Projects\\Face_recognition\\Data\\crop_part1"
    file_list = os.listdir(folder_path)
    np.random.shuffle(file_list)

    for i in file_list:
        img_path = os.path.join(folder_path,i)
        print(i)
        img = cv.imread(img_path)
        img = cv.resize(img, (200, 200))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        features = features.reshape(1,-1)
        dummy_columns = [f'f{i}' for i in range(features.shape[1])]
        features_df = pd.DataFrame(features, columns=dummy_columns)
        return features_df
        break

features_df = RealTime()
#features_df = random_photo()

age = model_age.predict(features_df)[0]
gender = model_gd.predict(features_df)[0]
race = model_race.predict(features_df)[0]
print(f"Predicted → Age: {age:.4f}, Gender: {gender}, Race: {race}")
