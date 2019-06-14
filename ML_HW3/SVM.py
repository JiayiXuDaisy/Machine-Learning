import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

def SVM(dataset):
    train_X = np.load('./data/feature_train-'+dataset+'.npy')
    train_y = np.load('./data/label_train-'+dataset+'.npy')
    test_X = np.load('./data/feature_test-'+dataset+'.npy')
    test_y = np.load('./data/label_test-'+dataset+'.npy')
    all_X = np.concatenate([train_X, test_X])
    if dataset == 'cod-rna':
        train_X = train_X[:1000]
        train_y = train_y[:1000]
        test_X = test_X[:4000]
        test_y = test_y[:4000]

    train_X = normalize(train_X, norm='max', axis=0, copy=True, return_norm=False)
    test_X = normalize(test_X, norm='max', axis=0, copy=True, return_norm=False)
    print(train_X.shape)
    print(test_X.shape)

    # test sample size
    '''
    if dataset == 'splice':
        for sample_size in [100, 200, 400, 800, 1000]:
            clf = SVC(kernel='linear')
            clf.fit(train_X[:sample_size], train_y[:sample_size])
            test_pred = clf.predict(test_X)
            accu = (test_pred == test_y).sum() / float(len(test_y))
            print(accu)
    elif dataset == 'cod-rna':
        # for sample_size in [10, 20, 50, 100, 1000, 2000, 4000, 8000, 16000, 24000, 32000, 48000, 59535]:
        for sample_size in [200,300,400,500,600,700,800,900]:
            clf = SVC(kernel='linear')
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
                train_X_red = all_X_red[:train_X.shape[0]]
                test_X_red = all_X_red[:test_X.shape[0]]
                clf = SVC(kernel='linear')
                clf.fit(train_X_red, train_y)
                test_pred = clf.predict(test_X_red)
                accu = (test_pred == test_y).sum() / float(len(test_y))
                print(accu)
            elif dimension == 60:
                clf = SVC(kernel='linear')
                clf.fit(train_X, train_y)
                test_pred = clf.predict(test_X)
                accu = (test_pred == test_y).sum() / float(len(test_y))
                print(accu)
    elif dataset == 'cod-rna':
        for dimension in [1,2,4,6,8]:
            if(dimension != 8):
                pca = PCA(n_components=dimension)
                all_X_red = pca.fit_transform(all_X)
                train_X_red = all_X_red[:train_X.shape[0]]
                test_X_red = all_X_red[:test_X.shape[0]]
                clf = SVC(kernel='linear')
                clf.fit(train_X_red, train_y)
                test_pred = clf.predict(test_X_red)
                accu = (test_pred == test_y).sum() / float(len(test_y))
                print(accu)
            elif dimension == 8:
                clf = SVC(kernel='linear')
                clf.fit(train_X, train_y)
                test_pred = clf.predict(test_X)
                accu = (test_pred == test_y).sum() / float(len(test_y))
                print(accu)
    '''

    # test kernel
    # '''
    print("test kernel for: "+str(dataset))
    for kernel in ['rbf', 'sigmoid']:
        clf = SVC(kernel=kernel)
        clf.fit(train_X, train_y)
        test_pred = clf.predict(test_X)
        accu = (test_pred == test_y).sum() / float(len(test_y))
        print(accu)
    # '''

    # test C
    '''
    for C in [0.01,0.03,0.06,0.1,0.3,0.6,1,3,6,10]:
        clf = SVC(kernel='linear',C=C)
        clf.fit(train_X, train_y)
        test_pred = clf.predict(test_X)
        accu = (test_pred == test_y).sum() / float(len(test_y))
        print(accu)
    '''

    # for sample_size in [100,200,400,800,1000]:
    #     for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    #         for C in [1e-3,1e-2,1e-1,1,2,4]:
    #             clf = SVC(kernel='linear')
    #             clf.fit(train_X[:sample_size], train_y[:sample_size])
    #             test_pred = clf.predict(test_X)
    #             accu = (test_pred == test_y).sum() / float(len(test_y))
    #             print('sample_size:'+str(sample_size)+'\tkernel:'+str(kernel)+'\tC:'+str(C)+'\nacc:'+str(accu))
    # all_X = np.concatenate([train_X,test_X])
    # print(all_X.shape)
    # for dimension in [1,2,4,6,8]:
    #     if dimension!=8:
    #         pca = PCA(n_components=dimension)
    #         all_X_red = pca.fit_transform(all_X)
    #         train_X_red = all_X_red[:train_X.shape[0]]
    #         test_X_red = all_X_red[:test_X.shape[0]]
    #         for kernel in ['linear','poly','rbf','sigmoid']:
    #             for C in [1e-3,1e-2,1e-1,1,2,4]:
    #                 clf = SVC(kernel=kernel,C=C)
    #                 clf.fit(train_X_red, train_y)
    #                 test_pred = clf.predict(test_X_red)
    #                 accu = (test_pred == test_y).sum() / float(len(test_y))
    #                 print('dimension:'+str(dimension)+'\tkernel:'+str(kernel)+'\tC:'+str(C)+'\nacc:'+str(accu))
    #     elif dimension == 8:
    #         for kernel in ['linear','poly','rbf','sigmoid']:
    #             for C in [1e-3,1e-2,1e-1,1,2,4]:
    #                 clf = SVC(kernel='linear')
    #                 clf.fit(train_X, train_y)
    #                 test_pred = clf.predict(test_X)
    #                 accu = (test_pred == test_y).sum() / float(len(test_y))
    #                 print('dimension:'+str(dimension)+'\tkernel:'+str(kernel)+'\tC:'+str(C)+'\nacc:'+str(accu))

if __name__ == '__main__':
    # SVM('splice')
    SVM('cod-rna')
