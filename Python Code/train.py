from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from model import *


def train(model):
    train_path = os.path.join("./flowers")
    train_data = create_train_data()
    train_history = train_model(model, train_data, epochs=10)
    model.save_weights(CHECKPOINT_FILE_NAME) #שומר את המשקלים
    show_graph_acc(train_history)
    show_graph_loss(train_history)


def create_train_data():
    # Generates images, its a tensorflow module
    # it applies different things on the images to make more images
    train_gen = ImageDataGenerator( # מגדילים את מאגר התמונות
        rescale=1. / 255.,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2
    )
    # Load train pictures from the directory (according to flower category)
    train_data = train_gen.flow_from_directory( # לוקח את התמונות מהנתיב בסוגריים
        train_path,
        target_size=(150, 150),
        batch_size=64,
        class_mode="categorical",  # tell the model how the output should be
        classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
        shuffle=True,
    )
    return train_data


def train_model(model, train_data, epochs=10):
    # train the model
    history = model.fit(train_data, epochs=epochs)
    return history


def show_graph_acc(history):
    history.history.keys()
    plt.plot(history.history['acc'])
    plt.title('model Accuracy ')
    plt.ylabel('accuracy ')
    plt.xlabel('epoch')
    plt.legend(['acc'], loc='upper left')
    plt.show()


def show_graph_loss(history):
    plt.plot(history.history['loss'])
    plt.title('model loss ')
    plt.ylabel('loss ')
    plt.xlabel('epoch')
    plt.legend(['loss'], loc='upper left')
    plt.show()