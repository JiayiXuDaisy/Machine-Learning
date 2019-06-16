import matplotlib.pyplot as plt
import numpy as np
import csv

labels = []
cln_data = []
with open("fer2013.csv", "r") as file:
    read = csv.reader(file)
    i = 0
    for line in read:
        if i:
            labels.append(line[0])
            cln = line[1].split(' ')
            data = []
            for c in cln:
                data.append(int(c))
            cln_data.append(data)
        i += 1


def draw():
    global labels
    global cln_data

    cln_d_data = []
    adv_data = []
    adv_d_data = []
    data = [cln_d_data, adv_data, adv_d_data]
    filenames = ["cln_defended.txt", "adversial.txt", "adv_defended.txt"]
    for i in range(3):
        with open(filenames[i], "r") as file:
            line = file.readline().split("\t")
            line = line[:-1]  # \n

            pixels = []
            n = 0
            while line:  # 48 * 48
                n += 1
                for j in range(len(line)):
                    line[j] = int(float(line[j]) * 255)
                pixels.append(line)
                line = file.readline().split("\t")
                line = line[:-1]
        data[i] = np.array(pixels)

    cln_d_data = data[0]
    adv_data = data[1]
    adv_d_data = data[2]

    cln_data = np.array(cln_data)
    cln_images = []
    cln_d_images = []
    adv_images = []
    adv_d_images = []
    for j in range(4):  # 3 a line
        cln_image = cln_data[j].reshape(48, 48)
        cln_images.append(cln_image)
        cln_d_image = cln_d_data[j].reshape(48, 48)
        cln_d_images.append(cln_d_image)
        adv_image = adv_data[j].reshape(48, 48)
        adv_images.append(adv_image)
        adv_d_image = adv_d_data[j].reshape(48, 48)
        adv_d_images.append(adv_d_image)

    plt.figure(figsize=(10, 10))
    for j in range(len(cln_images)):
        print(len(cln_images))
        plt.subplot(4, len(cln_images), j + 1)
        plt.axis('off')
        plt.imshow(cln_images[j], cmap="gray")
        plt.subplot(4, len(cln_images), j + 1 + len(cln_images))
        plt.axis('off')
        plt.imshow(cln_d_images[j], cmap="gray")
        plt.subplot(4, len(cln_images), j + 1 + 2 * len(cln_images))
        plt.axis('off')
        plt.imshow(adv_images[j], cmap="gray")
        plt.subplot(4, len(cln_images), j + 1 + 3 * len(cln_images))
        plt.axis('off')
        plt.imshow(adv_d_images[j], cmap="gray")
    plt.savefig("comparison.png")
    plt.show()


draw()
