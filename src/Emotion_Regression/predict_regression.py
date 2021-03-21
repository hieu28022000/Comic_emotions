from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np
import pandas as pd 
import glob2
from tqdm import tqdm

MODEL_SURPRISE = load_model('EfficientNetB4_image_regression_surprise.h5')
MODEL_SAD = load_model('EfficientNetB4_image_regression_sad.h5')
MODEL_OTHER = load_model('EfficientNetB4_image_regression_other.h5')
MODEL_NEUTRAL = load_model('EfficientNetB4_image_regression_neutral.h5')
MODEL_HAPPY = load_model('EfficientNetB4_image_regression_happy.h5')
MODEL_FEAR = load_model('EfficientNetB4_image_regression_fear.h5')
MODEL_DISGUST = load_model('EfficientNetB4_image_regression_disgust.h5')
MODEL_ANGRY = load_model('EfficientNetB4_image_regression_angry.h5')

def get_output_image_regression(image_id, image, input_size=380):
    image = cv2.resize(image, (input_size, input_size))
    image = image / 255.0
    image = image.reshape((1,input_size,input_size,3))

    prob_surprise = MODEL_SURPRISE.predict(image)[0][0] 
    prob_sad = MODEL_SAD.predict(image)[0][0] 
    prob_other = MODEL_OTHER.predict(image)[0][0]
    prob_neutral = MODEL_NEUTRAL.predict(image)[0][0] 
    prob_happy = MODEL_HAPPY.predict(image)[0][0]
    prob_fear = MODEL_FEAR.predict(image)[0][0] 
    prob_disgust = MODEL_DISGUST.predict(image)[0][0] 
    prob_angry = MODEL_ANGRY.predict(image)[0][0]

    result_dict = {
        'image_id': image_id,
        'angry': prob_angry,
        'disgust': prob_disgust,
        'fear': prob_fear,
        'happy': prob_happy,
        'sad': prob_sad,
        'surprise': prob_surprise,
        'neutral': prob_neutral,
        'other': prob_other
    }

    result_list = [image_id, prob_angry, prob_disgust, prob_fear, prob_happy, prob_sad, prob_surprise, prob_neutral, prob_other]

    return result_dict, result_list

def get_submit_image_regression(folder_dir):
    list_image_name = glob2.glob(os.path.join(folder_dir, '*.jpg'))
    len_list_image_name = len(list_image_name)
    list_output = []
    with tqdm(total=len_list_image_name) as pbar:
        for image_path in list_image_name:
            image = cv2.imread(image_path)
            image_id = (image_path.split("/")[-1]).split(".")[0]

            # get result
            result_dict, result_list = get_output_image_regression(image_id, image)
            list_output.append(result_list)
            pbar.update(1)
    
    df_output = pd.DataFrame(list_output)
    df_output.to_csv("output.csv")
    print("########## DONE #############")

if __name__ == '__main__':
    get_submit_image_regression('test')