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
                # old.append(frame)
                # fshape = frame.shape
                # fheight = fshape[0]
                # fwidth = fshape[1]
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

            # Use every 30 frames
            # for i in range(0, len(frames), 30):
            #     X.append(frames[i:np.min((i+30),len(frames))])
            #     y.append(value)

    X = np.array(X)
    y = np.array(y)

    return X, y

def train_video_model(data_dir, epochs=20, batch_size=4, model_weight='tahaWeights.h5'):
    # tf = # Add your TensorFlow import here
    model = videoFightModel(tf, is_train=True)
    print(model.summary())

    X, y = load_dataset(data_dir)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")

    # Data normalization
    # X_train = X_train / 255.0
    # X_val = X_val / 255.0

    # Model training
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    print(f"Training accuracy: {history.history['acc'][-1]}")
    print(f"Validation accuracy: {history.history['val_acc'][-1]}")

    # Save the trained weights
    model.save_weights(model_weight)
    print(f"Saved model weights to {model_weight}")

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# # Example usage:
# data_directory = "fight-detection-surv-dataset-master"
# train_video_model(data_directory)
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
                print(predaction[1])
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
            frames[i][:] = frm
            
            i+=1
        
        cv2.imshow('video', frame)
    

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

