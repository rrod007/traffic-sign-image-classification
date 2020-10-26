import traffic

import os
import sys
import tensorflow as tf
import cv2
import numpy as np


classification_dict = {
    0: "Maximum speed: 20", 1: "Maximum speed: 30", 2: "Maximum speed: 50",
    3: "Maximum speed: 60", 4: "Maximum speed: 70", 5: "Maximum speed: 80",
    6: "Speed de-restriction: 80", 7: "Maximum speed: 100", 8: "Maximum speed: 120",
    9: "No overtaking", 10: "No overtaking by trucks", 11: "Crossroad ahead, side roads to left and right",
    12: "Priority road ahead", 13: "Give way to all traffic", 14: "Stop and give way to all traffic",
    15: "Forbidden entry", 16: "Lorries - Trucks forbidden", 17: "No entry (one way traffic)",
    18: "Cars not allowed", 19: "Road ahead curves to the left side", 20: "Road ahead curves to the right side",
    21: "Road ahead bends left then right", 22: "Poor road surface ahead", 23: "Slippery road surface ahead",
    24: "Road gets narrow on the right side", 25: "Roadworks ahead", 26: "Traffic light ahead",
    27: "Warning for pedestrians", 28: "Warning for children and minors", 29: "Warning for bikes and cyclists",
    30: "Snow warning", 31: "Deer crossing in area - road", 32: "De-restriction sign",
    33: "Turning right compulsory", 34: "Turning left compulsory", 35: "Ahead only",
    36: "Driving ahead or right mandatory", 37: "Driving ahead or left mandatory", 38: "Pass on right compulsory",
    39: "Pass on left compulsory", 40: "Direction of traffic on roundabout", 41: "End of overtaking prohibition",
    42: "End of overtaking prohibition for trucks"
}


def predict():
    if len(sys.argv) != 3:
        sys.exit("Usage: python traffic.py model_to_use image_to_classify")

    # Load model
    model = tf.keras.models.load_model(sys.argv[1])

    # Load / resize image
    img = cv2.imread(sys.argv[2])
    img = cv2.resize(img, (traffic.IMG_WIDTH, traffic.IMG_HEIGHT))
    img = np.reshape(img, [1, traffic.IMG_WIDTH, traffic.IMG_HEIGHT, 3])

    # Throw image in model and get predicted class
    prediction = model.predict_classes(img)

    # Print classification
    print("Classification num: " + str(prediction[0]))
    print("Classification name: " + classification_dict[prediction[0]])


predict()
