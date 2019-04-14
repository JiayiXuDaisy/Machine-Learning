# Jiayi Xu 516021910396
# Machine Learning CS420
# Homework 1

import numpy as np
import random
import matplotlib.pyplot as plt

class DataGenerate(object):
    def __init__(self, class_num = 3, data_num = 100, center = None, dimension = 2):
        self.class_num = class_num
        self.data_num = data_num
        self.data = np.zeros((class_num * data_num,dimension), dtype = float)
        if center == None:
            self.center = [[np.random.uniform(0,class_num/1.3) for j in range(1,dimension+1)] for i in range(class_num)]
            # self.center = [(random.uniform(0,1),random.uniform(0,1)),(random.uniform(0,1),random.uniform(0,1)),(random.uniform(0,1),random.uniform(0,1))]
            # self.center = [(0,0),(0.5,1),(1,0.5)]
        else:
            self.center = center
        self.dimension =dimension


    def generate(self,name = None):
        for i in range(self.class_num):
            mu=np.zeros(self.dimension,dtype=float)
            sigma=np.zeros(self.dimension,dtype=float)
            for d in range(self.dimension):
                mu[d] = self.center[i][d]
                sigma[d] =  random.uniform(0.03*self.dimension,0.15*self.dimension)
                for j in range(self.data_num):
                    self.data[i * self.data_num + j][d] = np.random.normal(mu[d], sigma[d])

        if name == None:
            np.save("data.npy", self.data)
        else:
            np.save(name, self.data)

    def show(self):
        for i in range(self.class_num * self.data_num):
            x = self.data[i][0]
            y = self.data[i][1]
            plt.scatter(x,y,c = '#054E9F')
        plt.show()

if __name__ =='__main__':
    data = DataGenerate(dimension = 2)
    data.generate()
    data.show()

