import matplotlib.pyplot as plt
import os
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.test_utils import LeNet5
from advertorch_examples.utils import TRAINED_MODEL_PATH
from models import *


torch.manual_seed(0)
device = torch.device( "cpu")
net = VGG("VGG19")
checkpoint = torch.load('./models/PrivateTest_model.t7')
net.load_state_dict(checkpoint['net'])

print("start loading data!")
data = pd.read_csv("small_fer.csv")
print("loading data done")
pixels_values = data.pixels.str.split(" ").tolist()
print("split done")
pixels_values = pd.DataFrame(pixels_values, dtype=int)
print("change to narray done")
images = pixels_values.values
images = images.astype(np.float)

labels_flat = data["emotion"].values.ravel()
labels_flat = labels_flat

cln = []
for image in images:
    img = []
    image = image/255
    img_trans = image.reshape(48,48)
    img_trans = np.expand_dims(img_trans, axis=0)
    image = np.concatenate((img_trans, img_trans, img_trans), axis=0)
    cln.append(image)

cln_data = torch.FloatTensor(cln)
labels = torch.FloatTensor(labels_flat)

print(cln_data.shape)
print(labels.shape)

cln_data, labels = cln_data.to(device), labels.to(device,dtype=torch.int64)

from advertorch.attacks import LinfPGDAttack

adversary = LinfPGDAttack(net, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.15,
    nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
    targeted=False)

adv_untargeted = adversary.perturb(cln_data, labels)
adv = adv_untargeted

f = open('adversial.txt','a')
for data in adv_untargeted:
    dat = data[1].data.numpy().reshape(1,-1).ravel()
    # print(dat.shape)
    for d in dat:
        f.write('%.3f\t'%d)
    f.write('\n')


from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
from advertorch.defenses import JPEGFilter

bits_squeezing = BitSqueezing(bit_depth=5)
median_filter = MedianSmoothing2D(kernel_size=3)
jpeg_filter = JPEGFilter(10)

defense = nn.Sequential(
    jpeg_filter,
    bits_squeezing,
    median_filter,
)


adv_defended = defense(adv)
f = open('adv_defended.txt','a')
for data in adv_defended:
    dat = data[1].data.numpy().reshape(1,-1).ravel()
    # print(dat.shape)
    for d in dat:
        f.write('%.3f\t'%d)
    f.write('\n')
cln_defended = defense(cln_data)
f = open('cln_defended.txt','a')
for data in cln_defended:
    dat = data[1].data.numpy().reshape(1,-1).ravel()
    # print(dat.shape)
    for d in dat:
        f.write('%.3f\t'%d)
    f.write('\n')