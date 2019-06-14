import numpy as np
from sklearn.preprocessing import normalize
import random


def process(fname):
    label = []
    feature = []
    f = open("./data/"+fname+'.txt')
    rows = f.readlines()
    lines = []
    for row in rows:
        lines.append(row)
    random.shuffle(lines)
    for line in lines:
        line_ = line.split()
        label.append(line_.pop(0))
        f = []
        for l in line_:
            f.append(l.split(':')[1])
        feature.append(f)

    if fname == 'cod-rna':
        np.save('./data/label_'+fname+'.npy',label[:len(label)/10])
        np.save('./data/feature_'+fname+'.npy',feature[:len(feature)/10])
    else:
        np.save('./data/label_' + fname + '.npy', label[:len(label)])
        np.save('./data/feature_' + fname + '.npy', feature[:len(feature)])



if __name__ == '__main__':
    process('train-cod-rna')
    process('test-cod-rna')
    # process('train-splice')
    # process('test-splice')
    # label = np.load('./data/label_test-cod-rna.npy')
    # dict = {}
    # for key in label:
    #     dict[key] = dict.get(key, 0) + 1
    # print(dict)
    #normalize
    # feature = np.load('./data/feature_train-splice.npy')
    # print(feature)
    # feature = normalize(feature, norm='max', axis=0, copy=True, return_norm=False)
    # print(feature)
    # print(label.shape)
    # print(feature.shape)