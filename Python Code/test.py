import numpy as np
from keras.preprocessing import image as Kimage
import random
import matplotlib.pyplot as plt
from model import *


def test_model(model):
    # input: model
    # output: test the model and print the success rates
    # הפעולה בודקת את המוגל
    y_test = np.zeros((250, 5))  # make a result array for 250 images
    x_test = []  # images to test
    classes_list = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    class_dict = {"daisy": 0, "dandelion": 1, "rose": 2, "sunflower": 3, "tulip": 4}
    for clas in classes:  # go thru all the types of flowers
         if clas not in ["flowers"]:  # check only flower types and not the test folder
            images = os.listdir("./flowers" + "/" + clas)
            for i in range(50):
                random_name = random.choice(images)  # get random image from the array
                img = Kimage.load_img(os.path.join("./flowers" + "/" + clas + "/" + random_name),
                                      target_size=(150, 150))  # load the image
                x_test.append(np.array(img))  # add the image to the test and convert it to numbers for the model
                y_test[(class_dict[clas] * 50) + i][
                    class_dict[clas]] = 1  # give the computer the real result for this image
    x_test = np.asarray(x_test)  # prepare the array for the test
    x_test = x_test / 255.
    res = model.predict_classes(x_test)  # let the model predict the answers and get result
    actual = np.argmax(y_test, axis=-1)  # the real labels we gave it
    success = 0
    for i, j in zip(actual, res):  # go thru all the guess and results
        if i == j:
            success += 1
    print(f"Computer guessed it right {success} out of {len(actual)}  | {success / len(actual) * 100}% SUCCESS")


def test_one_picture(model, full_path, label):
    # input: model, the path of the image we want to check, label = name of the flower that we want check
    classes_list = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
    y_test = np.zeros((1, 5))  # create result array for the label given
    x_test = []  # the numerical value of the image
    img = Kimage.load_img(full_path, target_size=(150, 150))  # load the image from the path
    print("Showing selected image, close it to continue")
    plt.imshow(img)  # show the image that we check
    plt.show()
    x_test.append(np.array(img))  # convert the image to numbers, and append to the test array
    y_test[0][classes_list.index(label)] = 1  # tell the model that this is the right label (1 = chosen)
    x_test = np.asarray(x_test)  # prepare the test array for the model
    x_test = x_test / 255.
    res = model.predict_classes(x_test)  # give the model the image and let it guess the result
    model.evaluate(x_test, y_test, batch_size=1)  # train the model for that image
    actual = np.argmax(y_test, axis=-1)  # get the real result we told it
    if res == actual:  # if model was right
        print("The computer was right and guss that the flower is " + classes_list[int(res)] +
              " and the flower is actual " + classes_list[int(actual)])
    else:
        print("The computer was wrong")
