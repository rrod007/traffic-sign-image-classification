# Traffic Sign Image Recognition and Classification

This program, built using Python and Tensorflow, allows the user to train a keras Neural Network Sequential model on the German Traffic Sign Road Benchmark (GTSRB) dataset.
This dataset comprises images of traffic signs, which the model will be able to classify in categories after being trained.

## Requirements installation:
- You need to install Python itself
- Then, run the following command on the project directory:
```bash
pip install -r requirements.txt
```
Note: You do not need to install the "scikit-learn" package if you do not wish to train models.

## Example usage scenarios:

### Run predicitons on models pre-trained by me

- Second parameter: model
- Third parameter: image to classify
```bash
python predict.py trained_models/model2 try_images/class_14_img.ppm 
```

### Tweak NN layers and train models yourself

For this effect you should change the contents of the get_model() function.
Then, to train (and optionally save) the model:

- Second parameter: dataset to train on
- Third parameter (optional): path to save the model
```bash
python traffic.py gtsrb trained_models/model3
```


