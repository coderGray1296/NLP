import matplotlib.pyplot as plt
import numpy as np

def draw_picture(type, accuracy, loss):
    length = len(accuracy)
    x = np.arange(1, length+1, 1)
    plt.subplot(2, 1, 1)
    plt.plot(x, accuracy, "yo-")
    plt.title('accuracy of {}'.format(type))
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.grid()


    plt.subplot(2, 1, 2)
    plt.plot(x, loss, "r.-")
    plt.title('loss of {}'.format(type))
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid()
    plt.savefig("./pictures/{}.png".format(type))
    plt.show()