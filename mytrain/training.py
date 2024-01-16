from functions import *
import tensorflow as tf

def Training():

    tf.config.run_functions_eagerly(True)
    data_dir = "E:\\taha\\code\\Real-Time-Violence-Detection-in-Video-\\fight-detection-surv-dataset-master\\"
    model_weight='tahaWeights.h5'


    model = videoFightModel(tf, is_train=True)
    print(model.summary())

    X, y = load_dataset(data_dir)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Split the dataset into training and validatin sets
    X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=40)

    print(f"X_train shape: {X_train2.shape}")
    print(f"y_train shape: {y_train2.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")


    y_train2 = tf.one_hot(y_train2, depth=2)
    print(f'y_train shape: {y_train2.shape}')
    y_val = tf.one_hot(y_val, depth=2)
    print(f'y_val shape: {y_val.shape}')
    y_test = tf.one_hot(y_test, depth=2)
    print(f'y_test shape: {y_test.shape}')

    for i in range(11):
        # Model training
        history = model.fit(X_train2, y_train2, validation_data=(X_val, y_val), epochs=1, batch_size=6)
        print(f"Training accuracy: {history.history['accuracy'][-1]}")
        print(f"Validation accuracy: {history.history['val_accuracy'][-1]}")

        # Save the trained weights
        model.save_weights(model_weight)
        print(f"Saved model weights to {model_weight}")



