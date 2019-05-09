import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
import math
import csv
import matplotlib.pyplot as plt


class testFA(object):
    def sample_size(self):
        ############## test different sample size #################
        # m = 3
        # n = 4
        # noise level = 0.1
        mu = [0,0,0,0]
        mean_y = [0,0,0]
        cov_y = np.eye(3)
        mean_e = [0,0,0,0]
        cov_e = 0.1*np.eye(4)
        A = np.array([[8, 2, 1, 3], [5, 4, 9, 7], [7, 1, 7, 7]])

        score = np.empty([99,9])
        aic = np.empty([99,9])
        bic = np.empty([99,9])
        x_scale = []
        for index_i,sample_size in enumerate(range(10,1000,10)):
            print("***************************")
            print("sample size = "+ str(sample_size))
            x_scale.append(sample_size)
            y = np.random.multivariate_normal(mean_y, cov_y, sample_size)
            e = np.random.multivariate_normal(mean_e, cov_e, sample_size)
            x = np.dot(y,A) + mu + e
            for index_j, n_component in enumerate(range(1,10)):
                transformer = FactorAnalysis(n_components=n_component)
                y_pre = transformer.fit_transform(x)
                print("n_component = "+ str(n_component)+":")
                print("the average score")
                print(transformer.score(x))
                score[index_i][index_j] = sample_size*transformer.score(x)
                score_csv = pd.DataFrame(score)
                # score_csv.to_csv("./score_sample-size.csv", index=False, header=False)
                aic[index_i][index_j] = sample_size*transformer.score(x) - n_component
                aic_csv  = pd.DataFrame(aic)
                # aic_csv.to_csv("./aic_sample-size.csv",index=False,header=False)
                bic[index_i][index_j] = sample_size*transformer.score(x) - n_component/2*math.log(sample_size)
                bic_csv = pd.DataFrame(bic)
                # bic_csv.to_csv("./bic_sample-size.csv",index=False,header=False)

        for i in range(9):
            plt.plot(x_scale,score[:,i],label="n_component="+str(i+1))
        plt.legend()
        plt.title("Log-likelihood of FA with different n_components on different sample sizes")
        plt.xlabel("Sample size (N/10)")
        plt.ylabel("Log likelihood")
        plt.text(20, -100, "(n=4, m=3, noise level=0.1)", fontsize=10)
        plt.show()

        for i in range(9):
            plt.plot(x_scale,aic[:,i],label="n_component="+str(i+1))
        plt.legend()
        plt.title("Aic of FA with different n_components on different sample sizes")
        plt.xlabel("Sample size (N/10)")
        plt.ylabel("Aic score")
        plt.text(20, -100, "(n=4, m=3, noise level=0.1)", fontsize=10)
        plt.show()

        for i in range(9):
            plt.plot(x_scale,bic[:,i],label="n_component="+str(i+1))
        plt.legend()
        plt.title("Bic of FA with different n_components on different sample sizes")
        plt.xlabel("Sample size (N/10)")
        plt.ylabel("Bic score")
        plt.text(20, -100, "(n=4, m=3, noise level=0.1)", fontsize=10)
        plt.show()


        max_score = np.max(score, axis=1)
        max_score_index = np.argmax(score, axis=1)
        print(max_score)
        print(max_score_index)
        with open("./max_sample-size.csv", 'w', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(max_score_index)
            csv_write.writerow(max_score)
        max_aic = np.max(aic, axis=1)
        max_aic_index = np.argmax(aic, axis=1)
        print(max_aic)
        print(max_aic_index)
        with open("./max_sample-size.csv", 'a+', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(max_aic_index)
            csv_write.writerow(max_aic)
        max_bic = np.max(bic, axis=1)
        max_bic_index = np.argmax(bic, axis=1)
        print(max_bic)
        print(max_bic_index)
        with open("./max_sample-size.csv", 'a+', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(max_bic_index)
            csv_write.writerow(max_bic)

    def dim_n(self):
        ############## test different dimension n #################
        # sample size = 100
        # m = 3
        # noise level = 0.1
        mean_y = [0, 0, 0]
        cov_y = np.eye(3)

        score = np.empty([10, 9])
        aic = np.empty([10, 9])
        bic = np.empty([10, 9])

        x_scale = []
        for index_i, dim_n in enumerate(range(1, 20, 2)):
            x_scale.append(dim_n)
            print("***************************")
            print("dimension of n = " + str(dim_n))
            mu =np.zeros(dim_n)
            mean_e = np.zeros(dim_n)
            cov_e = 0.1 * np.eye(dim_n)
            e = np.random.multivariate_normal(mean_e, cov_e, 1000)
            y = np.random.multivariate_normal(mean_y, cov_y, 1000)
            print(y.shape)
            A = np.random.randint(1, 10, size=[3,dim_n])
            print(A.shape)
            x = np.dot(y, A) + mu + e
            for index_j, n_component in enumerate(range(1, 10)):
                transformer = FactorAnalysis(n_components=n_component)
                y_pre = transformer.fit_transform(x)
                print("n_component = " + str(n_component) + ":")
                print("the average score")
                print(transformer.score(x))
                score[index_i][index_j] = 1000* transformer.score(x)
                score_csv = pd.DataFrame(score)
                # score_csv.to_csv("./score_dim-n.csv", index=False, header=False)
                aic[index_i][index_j] =1000*transformer.score(x) - n_component *3
                aic_csv = pd.DataFrame(aic)
                # aic_csv.to_csv("./aic_dim-n.csv", index=False, header=False)
                bic[index_i][index_j] =1000* transformer.score(x) - n_component / 2 * math.log(1000) *3
                bic_csv = pd.DataFrame(bic)
                # bic_csv.to_csv("./bic_dim-n.csv", index=False, header=False)
        for i in range(9):
            plt.plot(x_scale,score[:,i],label="n_component="+str(i+1))
        plt.legend()
        plt.title("Log-likelihood of FA with different n_components on different n")
        plt.xlabel("Dimension of n")
        plt.ylabel("Log likelihood")
        plt.text(7.5, -5000, "(sample size = 1000, m=3, noise level=0.1)", fontsize=10)
        plt.show()

        for i in range(9):
            plt.plot(x_scale,aic[:,i],label="n_component="+str(i+1))
        plt.legend()
        plt.title("Aic of FA with different n_components on different n")
        plt.xlabel("Dimension of n")
        plt.ylabel("Aic score")
        plt.text(7.5, -5000, "(sample size = 1000, m=3, noise level=0.1)", fontsize=10)
        plt.show()

        for i in range(9):
            plt.plot(x_scale,bic[:,i],label="n_component="+str(i+1))
        plt.legend()
        plt.title("Bic of FA with different n_components on different n")
        plt.xlabel("Dimension of n")
        plt.ylabel("Bic score")
        plt.text(7.5, -5000, "(sample size = 1000, m=3, noise level=0.1)", fontsize=10)
        plt.show()

        max_score = np.max(score, axis=1)
        max_score_index = np.argmax(score, axis=1)
        print(max_score)
        print(max_score_index)
        with open("./max_dim-n.csv", 'w',newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(max_score_index)
            csv_write.writerow(max_score)
        max_aic = np.max(aic, axis=1)
        max_aic_index = np.argmax(aic,axis=1)
        print(max_aic)
        print(max_aic_index)
        with open("./max_dim-n.csv", 'a+', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(max_aic_index)
            csv_write.writerow(max_aic)
        max_bic = np.max(bic, axis=1)
        max_bic_index = np.argmax(bic, axis=1)
        print(max_bic)
        print(max_bic_index)
        with open("./max_dim-n.csv", 'a+', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(max_bic_index)
            csv_write.writerow(max_bic)

    def dim_m(self):
        ############## test different dimension n #################
        # sample size = 1000
        # n = 4,10,20,30
        # noise level = 0.1
        score = np.empty([10, 12])
        aic = np.empty([10, 12])
        bic = np.empty([10, 12])

        x_scale = [1,3,5,7,9,11,13,15,17,19,21,23]
        for index_i, dim_m in enumerate(range(1, 20, 2)):
            print("***************************")
            print("dimension of m = " + str(dim_m))
            mean_y = np.zeros(dim_m)
            cov_y = np.eye(dim_m)
            mu =np.zeros(20)
            mean_e = np.zeros(20)
            cov_e = 0.1 * np.eye(20)
            e = np.random.multivariate_normal(mean_e, cov_e, 1000)
            y = np.random.multivariate_normal(mean_y, cov_y, 1000)
            print(y.shape)
            A = np.random.randint(1, 10, size=[dim_m,20])
            print(A.shape)
            x = np.dot(y, A) + mu + e
            for index_j, n_component in enumerate(range(1, 25,2)):
                transformer = FactorAnalysis(n_components=n_component, max_iter = 10000)
                y_pre = transformer.fit_transform(x)
                print("n_component = " + str(n_component) + ":")
                print("the average score")
                print(transformer.score(x))
                score[index_i][index_j] = 1000 * transformer.score(x)
                score_csv = pd.DataFrame(score)
                # score_csv.to_csv("./score_dim-m_n-30.csv", index=False, header=False)
                aic[index_i][index_j] = 1000*transformer.score(x) - n_component
                aic_csv = pd.DataFrame(aic)
                # aic_csv.to_csv("./aic_dim-m_n-30.csv", index=False, header=False)
                bic[index_i][index_j] = 1000*transformer.score(x) - n_component / 2 * math.log(1000)
                bic_csv = pd.DataFrame(bic)
                # bic_csv.to_csv("./bic_dim-m_n-30.csv", index=False, header=False)
        for index,i in enumerate(range(1, 20, 2)):
            plt.plot(x_scale,score[index,:],label="actual m="+str(i))
        plt.legend(loc = "upper right")
        plt.title("Log-likelihood of FA with different n_components on different m")
        plt.xlabel("n_component")
        plt.ylabel("Log likelihood")
        plt.text(2.5, -15, "(sample size=1000, n=30, noise level=0.1)", fontsize=10)
        plt.show()

        for index,i in enumerate(range(1, 20, 2)):
            plt.plot(x_scale,aic[index,:],label="actual m="+str(i))
        plt.legend(loc = "upper right")
        plt.title("Aic of FA with different n_components on different m")
        plt.xlabel("n_component")
        plt.ylabel("Aic score")
        plt.text(2.5, -15, "(sample size=1000, n=30, noise level=0.1)", fontsize=10)
        plt.show()

        for index,i in enumerate(range(1, 20, 2)):
            plt.plot(x_scale,bic[index,:],label="actual m="+str(i))
        plt.legend(loc = "upper right")
        plt.title("Bic of FA with different n_components on different m")
        plt.xlabel("n_component")
        plt.ylabel("Bic score")
        plt.text(2.5, -20, "(sample size=1000, n=30, noise level=0.1)", fontsize=10)
        plt.show()

        max_score = np.max(score, axis=1)
        max_score_index = np.argmax(score, axis=1)
        print(max_score)
        print(max_score_index)
        with open("./max_dim-m_n-30.csv", 'w', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(max_score_index)
            csv_write.writerow(max_score)
        max_aic = np.max(aic, axis=1)
        max_aic_index = np.argmax(aic, axis=1)
        print(max_aic)
        print(max_aic_index)
        with open("./max_dim-m_n-30.csv", 'a+', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(max_aic_index)
            csv_write.writerow(max_aic)
        max_bic = np.max(bic, axis=1)
        max_bic_index = np.argmax(bic, axis=1)
        print(max_bic)
        print(max_bic_index)
        with open("./max_dim-m_n-30.csv", 'a+', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(max_bic_index)
            csv_write.writerow(max_bic)

    def noise(self):
        ############## test different dimension n #################
        # sample size = 1000
        # n = 4
        # m=3
        score = np.empty([20, 9])
        aic = np.empty([20, 9])
        bic = np.empty([20, 9])

        x_scale = []
        for index_i, noise_level in enumerate(range(0, 100, 5)):
            noise = noise_level*0.01
            x_scale.append(noise)
            print("***************************")
            print("noise level = " + str(noise))
            mean_y = np.zeros(3)
            cov_y = np.eye(3)
            mu =np.zeros(10)
            mean_e = np.zeros(10)
            cov_e = noise * np.eye(10)
            e = np.random.multivariate_normal(mean_e, cov_e, 1000)
            y = np.random.multivariate_normal(mean_y, cov_y, 1000)
            print(y.shape)
            A = np.random.randint(1, 10, size=[3,10])
            print(A.shape)
            x = np.dot(y, A) + mu + e
            for index_j, n_component in enumerate(range(1, 10,1)):
                transformer = FactorAnalysis(n_components=n_component, max_iter = 10000)
                y_pre = transformer.fit_transform(x)
                print("n_component = " + str(n_component) + ":")
                print("the average score")
                print(transformer.score(x))
                score[index_i][index_j] = 1000*transformer.score(x)
                score_csv = pd.DataFrame(score)
                # score_csv.to_csv("./score_noise.csv", index=False, header=False)
                aic[index_i][index_j] = 1000*transformer.score(x) - n_component
                aic_csv = pd.DataFrame(aic)
                # aic_csv.to_csv("./aic_noise.csv", index=False, header=False)
                bic[index_i][index_j] = 1000*transformer.score(x) - n_component / 2 * math.log(1000)
                bic_csv = pd.DataFrame(bic)
                # bic_csv.to_csv("./bic_noise.csv", index=False, header=False)
        print(score.shape)
        print(score[1,:].shape)
        for i in range(9):
            plt.plot(x_scale,score[:,i],label="n_component"+str(i+1))
        plt.legend(loc = "upper right")
        plt.title("Log-likelihood of FA with different n_components on different noise level")
        plt.xlabel("noise level")
        plt.ylabel("Log likelihood")
        plt.text(2.5, -10000, "(sample size=1000, n=10, m=3)", fontsize=10)
        plt.show()

        for i in range(9):
            plt.plot(x_scale,aic[:,i],label="n_component"+str(i+1))
        plt.legend(loc = "upper right")
        plt.title("Aic of FA with different n_components on different noise level")
        plt.xlabel("noise level")
        plt.ylabel("Aic score")
        plt.text(2.5, -10000, "(sample size=1000, n=10, m=3)", fontsize=10)
        plt.show()

        for i in range(9):
            plt.plot(x_scale,bic[:,i],label="n_component"+str(i+1))
        plt.legend(loc = "upper right")
        plt.title("Bic of FA with different n_components on different noise level")
        plt.xlabel("noise level")
        plt.ylabel("Bic score")
        plt.text(2.5, -10000, "(sample size=1000, n=10, m=3)", fontsize=10)
        plt.show()

        max_score = np.max(score, axis=1)
        max_score_index = np.argmax(score, axis=1)
        print(max_score)
        print(max_score_index)
        with open("./max_noise.csv", 'w', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(max_score_index)
            csv_write.writerow(max_score)
        max_aic = np.max(aic, axis=1)
        max_aic_index = np.argmax(aic, axis=1)
        print(max_aic)
        print(max_aic_index)
        with open("./max_noise.csv", 'a+', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(max_aic_index)
            csv_write.writerow(max_aic)
        max_bic = np.max(bic, axis=1)
        max_bic_index = np.argmax(bic, axis=1)
        print(max_bic)
        print(max_bic_index)
        with open("./max_noise.csv", 'a+', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(max_bic_index)
            csv_write.writerow(max_bic)

if __name__ == '__main__':
    test = testFA()
    # test.sample_size()
    # test.dim_n()
    # test.dim_m()
    # test.noise()