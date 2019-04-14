# Jiayi Xu 516021910396
# Machine Learning CS420
# Homework 1.3

from GMM import GMM
from VBEM import VBEM
from data_generate import DataGenerate
import numpy as np

def generate_sample_sizes():
    data_10 = DataGenerate(data_num=10)
    data_20 = DataGenerate(data_num=20)
    data_50 = DataGenerate(data_num=50)
    data_100 = DataGenerate(data_num=100)
    data_200 = DataGenerate(data_num=200)
    data_300 = DataGenerate(data_num=300)
    data_10.generate(name="./Data/data_10.npy")
    data_20.generate(name="./Data/data_20.npy")
    data_50.generate(name="./Data/data_50.npy")
    data_100.generate(name="./Data/data_100.npy")
    data_200.generate(name="./Data/data_200.npy")
    data_300.generate(name="./Data/data_300.npy")

def generate_cluster_num():
    cluster_2 = DataGenerate(class_num=2)
    cluster_3 = DataGenerate(class_num=3)
    cluster_4 = DataGenerate(class_num=4)
    cluster_5 = DataGenerate(class_num=5)
    cluster_6 = DataGenerate(class_num=6)
    cluster_2.generate(name="./Data/cluster_2.npy")
    cluster_3.generate(name="./Data/cluster_3.npy")
    cluster_4.generate(name="./Data/cluster_4.npy")
    cluster_5.generate(name="./Data/cluster_5.npy")
    cluster_6.generate(name="./Data/cluster_6.npy")

def generate_dimension():
    dimension_3 = DataGenerate(dimension=3)
    dimension_4 = DataGenerate(dimension=4)
    dimension_5 = DataGenerate(dimension=5)
    dimension_6 = DataGenerate(dimension=6)
    dimension_7 = DataGenerate(dimension=7)
    dimension_3.generate(name="./Data/dimension_3.npy")
    dimension_4.generate(name="./Data/dimension_4.npy")
    dimension_5.generate(name="./Data/dimension_5.npy")
    dimension_6.generate(name="./Data/dimension_6.npy")
    dimension_7.generate(name="./Data/dimension_7.npy")

def test_sample_sizes():
    for i in [10,20,50,100,200,300]:
        Data = np.load("./Data/data_"+str(i)+".npy")
        gmm = GMM(data = Data)
        gmm.aic()
        gmm.bic()
        vbem = VBEM(data = Data)
        vbem.train()
        vbem.show()

def test_cluster_num():
    for i in [2,3,4,5,6]:
        Data = np.load("./Data/cluster_"+str(i)+".npy")
        gmm = GMM(data = Data)
        gmm.aic()
        gmm.bic()
        vbem = VBEM(data = Data)
        vbem.train()
        vbem.show()

def test_dimension():
    for i in [3,4,5,6,7]:
        Data = np.load("./Data/dimension_"+str(i)+".npy")
        gmm = GMM(data = Data)
        gmm.aic()
        gmm.bic()
        vbem = VBEM(data = Data)
        vbem.train()
        vbem.show()

if __name__ == '__main__':
    # generate_sample_sizes()
    # test_sample_sizes()
    # generate_cluster_num()
    # test_cluster_num()
    generate_dimension()
    test_dimension()