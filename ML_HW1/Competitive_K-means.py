# Jiayi Xu 516021910396
# Machine Learning CS420
# Homework 1.2

from sklearn.cluster import KMeans
from data_generate import DataGenerate
import matplotlib.pyplot as plt
import numpy as np
import random
from math import sqrt, pow

data = np.load("data.npy")
class_num = 6
data_num = data.shape[0]
center = np.zeros((class_num,2),dtype = float)
r = np.zeros((data_num,class_num),dtype = int)
s = np.zeros((data_num,class_num),dtype = int)

# # original k-means with sklearn
# kmeans = KMeans(n_clusters = center_num).fit(data)
# center_pre = kmeans.cluster_centers_
# print(data.shape)
# print(center_pre)
# for i in range(data_num):
#     x = data[i][0]
#     y = data[i][1]
#     plt.scatter(x, y, c='#054E9F')
# for i in range(center_pre.shape[0]):
#     x = center_pre[i][0]
#     y = center_pre[i][1]
#     plt.scatter(x,y, c='red')
# plt.show()

# Competitive k-means by myself
cluster_changed = True

# initialize cluster centers
for i in range(class_num):
    center[i][0] = random.uniform(0,1)
    center[i][1] = random.uniform(0,1)

print(center.shape)

# EM for Competitive Kmeans
while cluster_changed:
    cluster_changed = False
    # E-step
    print("E-step")
    for i in range(data_num):
        minDis = 1000000
        index = 0
        for j in range(class_num):
            distance = np.sqrt(np.sum((data[i]-center[j])**2))
            if (distance < minDis):
                minDis = distance
                index = j
        if r[i][index] != 1:
            cluster_changed = True
            r[i,:] = np.zeros((1,class_num),dtype = int)
            r[i][index] = 1
    # M-step
    num = np.zeros((class_num,1),dtype = int)
    print("M-step")
    for i in range(class_num):
        pointInCluster = []
        for j in range(data_num):
            if r[j][i] == 1:
                pointInCluster.append(data[j,:])
                num[i] += 1
            if pointInCluster != []:
                center[i,:] = np.mean(pointInCluster,axis = 0)
    print(center)

    #RPCL step
    minDisCluster = 1000000
    index_1 = 0
    index_2 = 0
    for i in range(class_num):
        for j in range(i+1,class_num):
            distance = np.sqrt(np.sum((center[i]-center[j])**2))
            if distance < minDisCluster:
                index_1 = i
                index_2 = j
    if num[index_1] <= num[index_2]:
        center[index_1] = center[index_1] + 2 * (center[index_1] -center[index_2] )
    else:
        center[index_2] = center[index_2] + 2 * (center[index_2] - center[index_1])
    print(num)


# show the data points
for i in range(data_num):
    x = data[i][0]
    y = data[i][1]
    plt.scatter(x, y, c='#054E9F')

# show the center
for i in range(class_num):
    x = center[i][0]
    y = center[i][1]
    plt.scatter(x,y, c='red')

plt.axis([-1,2,-1,2])

plt.show()
