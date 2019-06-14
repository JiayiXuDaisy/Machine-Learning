from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

def MLP(dataset):
    train_X = np.load('./data/feature_train-' + dataset + '.npy')
    train_y = np.load('./data/label_train-' + dataset + '.npy')
    test_X = np.load('./data/feature_test-' + dataset + '.npy')
    test_y = np.load('./data/label_test-' + dataset + '.npy')
    train_X = normalize(train_X, norm='max', axis=0, copy=True, return_norm=False)
    test_X = normalize(test_X, norm='max', axis=0, copy=True, return_norm=False)
    tr_X = []
    te_X = []
    for i in train_X:
        tr_X.append(list(map(float, i)))
    train_X = tr_X
    train_y = list(map(int, train_y))
    for i in test_X:
        te_X.append(list(map(float,i)))
    test_X = te_X
    test_y = list(map(int,test_y))
    all_X = np.concatenate([train_X, test_X])
    print('('+str(len(train_X))+','+str(len(train_X[0]))+')')
    print('('+str(len(test_X)) + ',' + str(len(test_X[0]))+')')


    # test sample size
    '''
    if dataset == 'splice':
        for sample_size in [100, 200, 400, 800, 1000]:
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 15, 15), random_state=1)
            clf.fit(train_X[:sample_size], train_y[:sample_size])
            test_pred = clf.predict(test_X)
            accu = (test_pred == test_y).sum() / float(len(test_y))
            print(accu)
    elif dataset == 'cod-rna':
        for sample_size in [10, 20, 50, 100, 1000, 2000, 4000, 8000, 16000, 24000, 32000, 48000, 59535]:
        # for sample_size in [200,300,400,500,600,700,800,900]:
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15), random_state=1)
            clf.fit(train_X[:sample_size], train_y[:sample_size])
            test_pred = clf.predict(test_X)
            accu = (test_pred == test_y).sum() / float(len(test_y))
            print(accu)
    '''

    # test dimension
    '''
    if dataset == 'splice':
        for dimension in [1,2,4,8,16,24,32,40,48,60]:
            if(dimension != 60):
                pca = PCA(n_components=dimension)
                all_X_red = pca.fit_transform(all_X)
                train_X_red = all_X_red[:len(train_X)]
                test_X_red = all_X_red[:len(test_X)]
                clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15), random_state=1)
                clf.fit(train_X_red, train_y)
                test_pred = clf.predict(test_X_red)
                accu = (test_pred == test_y).sum() / float(len(test_y))
                print(accu)
            elif dimension == 60:
                clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15), random_state=1)
                clf.fit(train_X, train_y)
                test_pred = clf.predict(test_X)
                accu = (test_pred == test_y).sum() / float(len(test_y))
                print(accu)
    elif dataset == 'cod-rna':
        for dimension in [1,2,4,6,8]:
            if(dimension != 8):
                pca = PCA(n_components=dimension)
                all_X_red = pca.fit_transform(all_X)
                train_X_red = all_X_red[:len(train_X)]
                test_X_red = all_X_red[:len(test_X)]
                clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15), random_state=1)
                clf.fit(train_X_red, train_y)
                test_pred = clf.predict(test_X_red)
                accu = (test_pred == test_y).sum() / float(len(test_y))
                print(accu)
            elif dimension == 8:
                clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15), random_state=1)
                clf.fit(train_X, train_y)
                test_pred = clf.predict(test_X)
                accu = (test_pred == test_y).sum() / float(len(test_y))
                print(accu)
    '''

    # test alpha
    for alpha in [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]:
        clf = MLPClassifier(solver='lbfgs', alpha=alpha, hidden_layer_sizes=(15), random_state=1)
        clf.fit(train_X, train_y)
        test_pred = clf.predict(test_X)
        accu = (test_pred == test_y).sum() / float(len(test_y))
        print(accu)

if __name__ == '__main__':
    MLP('splice')
    MLP('cod-rna')