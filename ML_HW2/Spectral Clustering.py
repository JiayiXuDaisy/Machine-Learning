from sklearn.cluster import SpectralClustering
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import warnings
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice



class test_sc(object):
    def guassian(self):
        color = ['LightPink', 'HotPink', 'MediumPurple', 'MediumSlateBlue', 'RoyalBlue', 'SlateGray', 'SkyBlue',
                              'DarkCyan', 'MediumAquamarine', 'LightGreen', 'DarkKhaki', 'Tan', 'Chocolate', 'Tomato', 'Brown']

        class_num = 3
        data_num = 1000
        dimension = 6
        center =  [[np.random.uniform(0,2) for j in range(1,dimension+1)] for i in range(class_num)]
        data = np.zeros((class_num * data_num,dimension), dtype = float)
        for i in range(class_num):
            mu = np.zeros(dimension, dtype=float)
            sigma = np.zeros(dimension, dtype=float)
            for d in range(dimension):
                mu[d] = center[i][d]
                sigma[d] = random.uniform(0.03 * dimension, 0.15 * dimension)
                for j in range(data_num):
                   data[i * data_num + j][d] = np.random.normal(mu[d], sigma[d])

        for i in range(class_num):
            for j in range(data_num):
                x = data[i*data_num + j][0]
                y = data[i*data_num + j][1]
                plt.scatter(x,y,c=color[i])
        plt.savefig('./class_'+str(class_num)+'_data_'+str(data_num)+'_dim_'+str(dimension)+'.png')
        plt.show()
        print(data)

        n_cluster = 3
        clustering = SpectralClustering(n_clusters=n_cluster, assign_labels="discretize",random_state=0).fit(data)
        print(clustering.labels_)
        label = clustering.labels_
        accuracy = 0
        for i in range(class_num):
            label_count = np.zeros([3, 1])
            for j in range(data_num*i,data_num*(i+1)):
                label_count[label[j]] += 1
            print(label_count)
            label_max_count = np.max(label_count, axis=0)
            print(label_max_count)
            accuracy += label_max_count/data_num
        accuracy = accuracy/class_num
        print(accuracy)

        for i in range(class_num):
            for j in range(data_num):
                x = data[i * data_num + j][0]
                y = data[i * data_num + j][1]
                plt.scatter(x, y, c=color[label[i* data_num + j]])
        plt.title("clustering accuracy = "+str(accuracy))
        plt.savefig('./class_' + str(class_num) + '_data_' + str(data_num) + '_dim_' + str(dimension) + '_pre.png')
        plt.show()
    def blob(self):
        color = ['LightPink', 'HotPink', 'MediumPurple', 'MediumSlateBlue', 'RoyalBlue', 'SlateGray', 'SkyBlue',
                 'DarkCyan', 'MediumAquamarine', 'LightGreen', 'DarkKhaki', 'Tan', 'Chocolate', 'Tomato', 'Brown']
        class_num =5
        data_num = 10
        dimension = 2
        data,label = datasets.make_blobs(n_samples=data_num*class_num, centers = class_num,n_features= dimension)

        print(label)
        for i in range(data.shape[0]):
                x = data[i][0]
                y = data[i][1]
                plt.scatter(x,y,c=color[label[i]])
        plt.savefig('./blob_class_' + str(class_num) + '_data_' + str(data_num) + '_dim_' + str(dimension) + '.png')
        plt.show()

        n_cluster = 3
        clustering = SpectralClustering(n_clusters=n_cluster, assign_labels="discretize", random_state=0).fit(data)
        print(clustering.labels_)
        label_pre = clustering.labels_

        id_0 = [j for j, x in enumerate(label) if x == 0]
        id_1 = [j for j, x in enumerate(label) if x == 1]
        id_2 = [j for j, x in enumerate(label) if x == 2]

        accuracy = 0
        label_count = np.zeros([3, 1])
        for i, x in enumerate(id_0):
            label_count[label_pre[x]] += 1
        label_max_count = np.max(label_count, axis=0)
        accuracy += label_max_count / data_num
        print(label_count)

        label_count = np.zeros([3, 1])
        for i, x in enumerate(id_1):
            label_count[label_pre[x]] += 1
        label_max_count = np.max(label_count, axis=0)
        accuracy += label_max_count / data_num
        print(label_count)

        label_count = np.zeros([3, 1])
        for i, x in enumerate(id_2):
            label_count[label_pre[x]] += 1
        label_max_count = np.max(label_count, axis=0)
        accuracy += label_max_count / data_num
        print(label_count)

        accuracy =accuracy/class_num
        print(accuracy)

        for i in range(data.shape[0]):
                x = data[i][0]
                y = data[i][1]
                plt.scatter(x,y,c=color[label[i]])
        plt.title("accuracy = " + str(accuracy))
        plt.savefig('./blob_class_' + str(class_num) + '_data_' + str(data_num) + '_dim_' + str(dimension) + '_pre.png')
        plt.show()
    def toy(self):
        np.random.seed(0)
        from sklearn import datasets
        # ============
        # Generate datasets. We choose the size big enough to see the scalability
        # of the algorithms, but not too big to avoid too long running times
        # ============
        n_samples = 1500
        noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                              noise=.05)
        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
        blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
        no_structure = np.random.rand(n_samples, 2), None

        # Anisotropicly distributed data
        random_state = 170
        X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        aniso = (X_aniso, y)

        # blobs with varied variances
        varied = datasets.make_blobs(n_samples=n_samples,
                                     cluster_std=[1.0, 2.5, 0.5],
                                     random_state=random_state)

        # ============
        # Set up cluster parameters
        # ============
        plt.figure(figsize=(9 * 2 + 3, 20))
        plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                            hspace=.01)

        plot_num = 1

        default_base = {'quantile': .3,
                        'eps': .3,
                        'damping': .9,
                        'preference': -200,
                        'n_neighbors': 10,
                        'n_clusters': 3}

        datasets = [
            (noisy_circles, {'damping': .77, 'preference': -240,
                             'quantile': .2, 'n_clusters': 2}),
            (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
            (varied, {'eps': .18, 'n_neighbors': 2}),
            (aniso, {'eps': .15, 'n_neighbors': 2}),
            (blobs, {}),
            (no_structure, {})]

        for i_dataset, (dataset, algo_params) in enumerate(datasets):
            # update parameters with dataset-specific values
            params = default_base.copy()
            params.update(algo_params)

            X, y = dataset

            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)

            # estimate bandwidth for mean shift
            bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])

            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(
                X, n_neighbors=params['n_neighbors'], include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)

            # ============
            # Create cluster objects
            # ============
            ms = cluster.SpectralClustering(
                n_clusters=params['n_clusters'], eigen_solver='arpack',
                affinity="nearest_neighbors")
            spectral = cluster.SpectralClustering(
                n_clusters=params['n_clusters'], eigen_solver='arpack',
                affinity="nearest_neighbors",assign_labels='kmeans')
            dbscan = cluster.SpectralClustering(
                n_clusters=params['n_clusters'], eigen_solver='arpack',
                affinity="nearest_neighbors",assign_labels='discretize')

            clustering_algorithms = (
                ('eigen_solver=arpack', ms),
                ('assign_labels=kmeans', spectral),
                ('assign_labels=discretize', dbscan)
            )

            for name, algorithm in clustering_algorithms:
                t0 = time.time()

                # catch warnings related to kneighbors_graph
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components of the " +
                                "connectivity matrix is [0-9]{1,2}" +
                                " > 1. Completing it to avoid stopping the tree early.",
                        category=UserWarning)
                    warnings.filterwarnings(
                        "ignore",
                        message="Graph is not fully connected, spectral embedding" +
                                " may not work as expected.",
                        category=UserWarning)
                    algorithm.fit(X)

                t1 = time.time()
                if hasattr(algorithm, 'labels_'):
                    y_pred = algorithm.labels_.astype(np.int)
                else:
                    y_pred = algorithm.predict(X)

                plt.subplot(len(datasets), len(clustering_algorithms), plot_num)
                if i_dataset == 0:
                    plt.title(name, size=18)

                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']),
                                              int(max(y_pred) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

                plt.xlim(-2.5, 2.5)
                plt.ylim(-2.5, 2.5)
                plt.xticks(())
                plt.yticks(())
                plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                         transform=plt.gca().transAxes, size=15,
                         horizontalalignment='right')
                plot_num += 1

        plt.show()


if __name__ == '__main__':
    test = test_sc()
    # test.guassian()
    # test.blob()
    test.toy()

