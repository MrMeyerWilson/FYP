import FYP2_code as fyp
import numpy as np
import pickle
import os

def classify(image_path, image_mask_path):
    width, height, diameter, perimeter_pixel, area, compactness, symmetry_pixel, colors_cancerous = fyp.get_features(image_path, image_mask_path)
    features = [width, height, diameter, perimeter_pixel, area, compactness, symmetry_pixel]
    features.extend(list(colors_cancerous[0]))
    features.extend(list(colors_cancerous[1]))
    features.extend(list(colors_cancerous[2]))
    features = np.reshape(features, (1, -1))

    classifier = pickle.load(open("Final_Model5", "rb"))
    label = classifier.predict(features)
    probability = classifier.predict_proba(features)
    return label, probability
    
def main():
    cdir = os.getcwd()
    sample_image = f"{cdir}/PAT_191_294_629.png"
    sample_mask = f"{cdir}/PAT_191_294_629_Mask.png"
    print(classify(sample_image, sample_mask))
    
if __name__ == "__main__":
    main()