# Jiayi Xu 516021910396
# Machine Learning CS420
# Homework 1.3

from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

class GMM(object):
    def __init__(self,n_components = 16, covariance_type='diag', random_state=0, data = [] ):
        self.color = ['LightPink', 'HotPink', 'MediumPurple', 'MediumSlateBlue', 'RoyalBlue', 'SlateGray', 'SkyBlue',
                      'DarkCyan', 'MediumAquamarine', 'LightGreen', 'DarkKhaki', 'Tan', 'Chocolate', 'Tomato', 'Brown']
        self.n_components = n_components
        if data == []:
            self.data = np.load("data.npy")
        else:
            self.data = data

    # select k through AIC
    def aic(self):
        AIC = np.zeros(self.n_components - 1, dtype=float)
        for n in range(1, self.n_components):
            clf = GaussianMixture(n_components=n, covariance_type='diag', random_state=0)
            clf.fit(self.data)
            AIC[n - 1] = clf.aic(self.data)
        # print(AIC)
        aic_min_index = np.where(AIC == np.min(AIC))
        aic_k = aic_min_index[0][0] + 1
        print("The number of cluster centers AIC choose is :" + str(aic_k))

        aic_gmm = GaussianMixture(n_components= aic_k, covariance_type='diag', random_state=0)
        aic_gmm.fit(self.data)
        labels = aic_gmm.predict(self.data)
        for i in range(1, len(labels)):
            for j in range(aic_k):
                if labels[i] == j :
                    plt.scatter(self.data[i][0], self.data[i][1], s=15, c=self.color[j])
        plt.title('GMM-AIC-'+str(len(labels)))
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.savefig("./Fig/sample_size_aic_" + str(len(labels)) + ".png")
        # plt.savefig("./Fig/cluster_num_aic_" + str(len(labels)) + ".png")
        # plt.savefig("./Fig/dimension_num_aic_" + str(len(self.data[0])) + ".png")
        plt.show()
        # print(aic_gmm.means_)
        # print(aic_gmm.covariances_)

    # select K through BIC
    def bic(self):
        BIC = np.zeros(self.n_components - 1, dtype=float)
        for n in range(1, self.n_components):
            clf = GaussianMixture(n_components=n, covariance_type='diag', random_state=0)
            clf.fit(self.data)
            labels = clf.predict(self.data)
            BIC[n - 1] = clf.bic(self.data)
        # print(BIC)
        bic_min_index = np.where(BIC == np.min(BIC))
        bic_k = bic_min_index[0][0] + 1
        print("The number of cluster centers BIC choose is :" + str(bic_k))

        bic_gmm = GaussianMixture(n_components=bic_k, covariance_type='diag', random_state=0)
        bic_gmm.fit(self.data)
        labels = bic_gmm.predict(self.data)
        for i in range(1, len(labels)):
            for j in range(bic_k):
                if labels[i] == j:
                    plt.scatter(self.data[i][0], self.data[i][1], s=15, c=self.color[j])
        plt.title('GMM-BIC-'+str(len(labels)))
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.savefig("./Fig/sample_size_bic_" + str(len(labels)) + ".png")
        # plt.savefig("./Fig/cluster_num_bic_" + str(len(labels)) + ".png")
        # plt.savefig("./Fig/dimension_num_bic_" + str(len(self.data[0])) + ".png")
        plt.show()
        # print(bic_gmm.means_)
        # print(bic_gmm.covariances_)

if __name__ == '__main__':
    gmm = GMM()
    gmm.aic()
    gmm.bic()