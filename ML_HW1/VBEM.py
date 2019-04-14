# Jiayi Xu 516021910396
# Machine Learning CS420
# Homework 1.3

from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import matplotlib.pyplot as plt

class VBEM(object):
    def __init__(self, n_components = 16, covariance_type = 'diag',random_state = 0,data = []):
        self.color = ['LightPink','HotPink','MediumPurple','MediumSlateBlue','RoyalBlue','SlateGray','SkyBlue','DarkCyan','MediumAquamarine','LightGreen','DarkKhaki','Tan','Chocolate','Tomato','Brown','Maroon']
        self.n_components = n_components
        if data == []:
            data = np.load("data.npy")
        else:
            data = data
        self.data = data
        self.model = BayesianGaussianMixture(n_components = n_components, covariance_type='diag',random_state=0)

    def train(self):
        self.model.fit(self.data)

    def show(self):
        labels = self.model.predict(self.data)
        # print(labels)
        # print(set(labels))
        for i in range(1,len(labels)):
            for j in range(self.n_components):
                if labels[i] == j :
                    plt.scatter(self.data[i, 0], self.data[i, 1], s=15, c=self.color[j])
        plt.title('VBEM-'+str(len(labels)))
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.savefig("./Fig/sample_size_vbem_"+str(len(labels))+".png")
        # plt.savefig("./Fig/cluster_num_vbem_" + str(len(labels)) + ".png")
        # plt.savefig("./Fig/dimension_num_vbem_" + str(len(self.data[0])) + ".png")
        plt.show()
        print("The number of cluster centers VBEM choose is :" + str(len(set(labels))))
        # print(self.model.means_)
        # print(self.model.covariances_)
        # print(self.model.lower_bound_)

if __name__ == '__main__':
    vbem = VBEM()
    vbem.train()
    vbem.show()