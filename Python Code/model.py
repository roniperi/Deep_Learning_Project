import os
from keras import layers, models


CHECKPOINT_FILE_NAME = "mycheckpoint" #  משתנה שבו שומרים את המשקלים
train_path = os.path.join("./flowers")
classes = os.listdir(train_path)


def load_model():
    model = create_model()
    try:
        model.load_weights(CHECKPOINT_FILE_NAME)  # try to load train history מנסה לראות אם יש משקלים שמורים
    except Exception:  # if its the first run the file does not exist, ignore it אם אין משתמש במודל ומחזיר אותו
        print("can't load weights yet")
    return model


def create_model():
    model = models.Sequential()
    # first layer would always be the input layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    # next layers are the hidden layers
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    # the last layer is the output layer
    model.add(layers.Dense(5, activation='softmax'))
    # build the model
    model.compile(loss="categorical_crossentropy", metrics="acc", optimizer='adam')
    return model
