import os
import cv2 as cv
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor,LGBMClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import joblib
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
import pandas as pd
# import warnings
# warnings.filterwarnings("ignore")

''' Loads The database 
W = image
X = Age
Y = Gender
Z = Race 
'''

age_model_path = "D:\\Anshul\\Anshul Codes\\Projects\\Semester Projects\\Face_recognition\\Models\\model_age.pkl"
gender_model_path = "D:\\Anshul\\Anshul Codes\\Projects\\Semester Projects\\Face_recognition\\Models\\model_gd.pkl"
race_model_path = "D:\\Anshul\\Anshul Codes\\Projects\\Semester Projects\\Face_recognition\\Models\\model_race.pkl"

def Load_Database():
    folder_path = "D:\\Anshul\\Anshul Codes\\Projects\\Semester Projects\\Face_recognition\\Data\\crop_part1"
    image = []
    age = []
    race = []
    gender = []
    count = 0
    file_list = os.listdir(folder_path)
    np.random.shuffle(file_list)

    for i in file_list:
        img_path = os.path.join(folder_path,i)
        img = cv.imread(img_path)
        if img is None:
            continue
        try:
            data = i.split("_")
            a = int(data[0])
            g = int(data[1])
            r = int(data[2])

            img = cv.resize(img, (200, 200))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)

            image.append(features)
            age.append(a)
            gender.append(g)
            race.append(r)

            count += 1
            if count % 500 == 0:
                print(f"{count} images processed...")

        except (ValueError, IndexError):
            continue

    image = pd.DataFrame(image)
    print(f"Total Image processed = {count}")
    return image , age , gender , race

w,x,y,z = Load_Database()
print(len(w), len(x), len(y), len(z))

def Train_Model(img , age , gender , race):
    img_train, img_test, age_train, age_test, gd_train, gd_test, race_train, race_test = train_test_split(img, age, gender, race, test_size=0.2, random_state=42)
    print("spilition complete")

    model_age = LGBMRegressor(n_estimators=100, max_depth=6,min_data_in_leaf=10,num_leaves=31,verbose=-1,n_jobs=-1,random_state=42)
    model_race = LGBMClassifier(n_estimators=100, max_depth=6,min_data_in_leaf=10,num_leaves=31,verbose=-1,n_jobs=-1,random_state=42)
    model_gd = LGBMClassifier(n_estimators=100, max_depth=6,min_data_in_leaf=10,num_leaves=31,verbose=-1,n_jobs=-1,random_state=42)
    print("model created")

    model_gd.fit(img_train,gd_train)
    print("gd model trained")
    gd_pred = model_gd.predict(img_test)
    m_gd = accuracy_score(gd_test, gd_pred)
    print("Gender accuracy = ",m_gd)
    joblib.dump(model_gd,gender_model_path)
    print("Gender model saved")

    model_race.fit(img_train,race_train)
    print("race model trained")
    race_pred = model_race.predict(img_test)    
    m_race = accuracy_score(race_test, race_pred)
    print("Race accuracy = ",m_race)
    joblib.dump(model_race,race_model_path)
    print("race model saved")

    model_age.fit(img_train,age_train)
    print("age model trained")
    age_pred = model_age.predict(img_test)
    m_age = mean_absolute_error(age_test, age_pred)
    print("Age accuracy = ",m_age)
    joblib.dump(model_age,age_model_path)
    print("age model saved")
    
    return model_age,model_gd,model_race

model_age , model_gd , model_race  = Train_Model(w,x,y,z)
#model_age = Train_Model(w,x,y,z)

