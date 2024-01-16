import numpy as np
from skimage.transform import resize
import tensorflow as tf
import os
import glob
import cv2

def videoFightModel(tf=tf,wight='tahaWeights.h5',is_train=False):
    layers = tf.keras.layers
    models = tf.keras.models
    losses = tf.keras.losses
    optimizers = tf.keras.optimizers
    metrics = tf.keras.metrics
    num_classes = 2
    cnn = models.Sequential()
    #cnn.add(base_model)

    input_shapes=(160,160,3)
    np.random.seed(1234)
    vg19 = tf.keras.applications.vgg19.VGG19
    base_model = vg19(include_top=False,weights='imagenet',input_shape=(160, 160,3))
    # Freeze the layers except the last 4 layers
    for layer in base_model.layers:
       layer.trainable = False

    cnn = models.Sequential()
    cnn.add(base_model)
    cnn.add(layers.Flatten())
    model = models.Sequential()

    model.add(layers.TimeDistributed(cnn,  input_shape=(30, 160, 160, 3)))
    model.add(layers.LSTM(30 , return_sequences= True))

    model.add(layers.TimeDistributed(layers.Dense(90)))
    model.add(layers.Dropout(0.1))

    model.add(layers.GlobalAveragePooling1D())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation="sigmoid"))

    adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    
    if not is_train:
        model.load_weights(wight)
        print(f"model loaded from: {wight}")
    rms = optimizers.RMSprop()

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

    return model

def pred_fight(model,video,acuracy=0.9):
    pred_test = model.predict(video)
    if pred_test[0][1] >=acuracy:
        return True , pred_test[0][1]
    else:
        return False , pred_test[0][1]
    

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_dataset(data_dir):
    X = []
    y = []
    class_labels = {'fight': 1, 'noFight': 0}

    print(f"Loading {data_dir} dataset...")

    for label, value in class_labels.items():
        label_path = os.path.join(data_dir, label) +"\\"
        for filename in os.listdir(label_path):
            video_path = os.path.join(label_path, filename)
            cap = cv2.VideoCapture(video_path)
            frames = np.zeros((30, 160, 160, 3), dtype=float)
            j = 1
            i = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frm = resize(frame, (160, 160, 3))
                frm = np.expand_dims(frm,axis=0)
                if(np.max(frm)>1):
                    frm = frm/255.0
                frames[i][:] = frm
                if j > 29:
                    X.append(frames)
                    y.append(value)
                    j = 0
                    i = -1
                    frames = np.zeros((30, 160, 160, 3), dtype=float)
                j += 1
                i += 1
            cap.release()


    X = np.array(X)
    y = np.array(y)

    return X, y

def test_model(model, X_test, y_test):
    print("Testing the model...")
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=4)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")

def stream(filename, model_dir):
    model = videoFightModel(tf,wight=model_dir)

    cap = cv2.VideoCapture(filename)
    i = 0
    frames = np.zeros((30, 160, 160, 3), dtype=float)
    old = []
    j = 0
    while(True):
        ret, frame = cap.read()
        if frame is None:
            print("end of the video")
            break
    
        # describe the type of font
        # to be used.
        font = cv2.FONT_HERSHEY_SIMPLEX
        if i > 29:
            ysdatav2 = np.zeros((1, 30, 160, 160, 3), dtype=float)
            ysdatav2[0][:][:] = frames
            predaction = pred_fight(model,ysdatav2,acuracy=0.967355)

            if predaction[0] == True:
                # print(predaction[1])
                cv2.putText(frame, 
                    'Violance Deacted', 
                    (50, 50), 
                    font, 3, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4)
                cv2.imshow('video', frame)
                print('Violance detacted here ...')

            i = 0
            j += 1
            frames = np.zeros((30, 160, 160, 3), dtype=float)
            old = []
        else:
            frm = resize(frame,(160,160,3))
            old.append(frame)
            fshape = frame.shape
            fheight = fshape[0]
            fwidth = fshape[1]
            frm = np.expand_dims(frm,axis=0)
            if(np.max(frm)>1):
                frm = frm/255.0

            #  frame skipping. set j%1 to use all frames
            if j%2 == 0:
                frames[i][:] = frm
                
                i+=1
        
        cv2.imshow('video', frame)
        j += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

