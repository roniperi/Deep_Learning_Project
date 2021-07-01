import tkinter
from train import *
from test import *
from model import *


train_path = os.path.join("./flowers")  # the path to the stock Images
classes = os.listdir(train_path)  # Represents the five folders of the flower types

window = tkinter.Tk()  # create the window of the buttons
window.title("Roni Project")


def start_menu():
    # input: none
    # output: window with buttons
    top_frame = tkinter.Frame(window).pack()
    bottom_frame = tkinter.Frame(window).pack(side="bottom")
    window.geometry('400x400')  # window size
    model = load_model()
    tkinter.Label(text="Hello, please choose an option:").pack()
    tkinter.Button(window, text='show initial info', activeforeground="black",
                   activebackground="pink",
                   pady=10, command=lambda: show_initial_info()).pack(pady=10)
    tkinter.Button(window, text='train the model + show loss and accuracy graphs', activeforeground="black", activebackground="red",
                   pady=10, command=lambda: train(model)).pack(pady=10)
    tkinter.Button(window, text='test the model', activeforeground="black", activebackground="blue",
                   pady=10, command=lambda: test_model(model)).pack(pady=10)
    tkinter.Button(window, text='Test new image', activeforeground="black",
                   activebackground="green", pady=10, command=lambda: input_for_test_image(model)).pack(pady=10)
    window.mainloop()


def input_for_test_image(model):
    full_path = input("Please enter the image path: ")
    label = input("please enter the name of the flower:")
    test_one_picture(model, full_path, label)


def show_initial_info():
    count = 0
    for clas in classes:
        count += len(os.listdir(train_path + "/" + clas))
    print("total flowers: ", count)
    print("total flower types: ", classes)


def main():
    start_menu()


if __name__ == '__main__':
    main()
