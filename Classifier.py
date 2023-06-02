import numpy as np
import pickle
import Feature_Extractor as f_extract

def classify(image_path, image_mask_path):
    width, height, diameter, perimeter_pixel, area, compactness, assymetry_pixel, colors_cancerous = f_extract.get_features(image_path, image_mask_path)
    
    features = [width, height, diameter, perimeter_pixel, area, compactness, assymetry_pixel]
    features.extend(colors_cancerous[0])
    features.extend(colors_cancerous[1])
    features.extend(colors_cancerous[2])
    features = np.reshape(features, (1, -1))
    
    classifier = pickle.load(open("Final_Model5", "rb")) 
    probability = classifier.predict_proba(features)
    label = classifier.predict(features)
    
    return label, probability
    

def main():
    sample_imagee = "PAT_191_294_629.png"
    sample_mask = "PAT_191_294_629_Mask.png"
    print(classify(sample_imagee, sample_mask))

if __name__ == "__main__":
    main()