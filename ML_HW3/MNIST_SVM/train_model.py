from input_data import get_labels,get_images
from sklearn import svm
import pickle
import numpy as np

print("start")
train_data = get_images('./Input_Data/train-images-idx3-ubyte/train-images.idx3-ubyte', length=50000)
print("get train data done")
train_labels = get_labels('./Input_Data/train-labels-idx1-ubyte/train-labels.idx1-ubyte')
print("get train label done")

print("reshape data")
train_data = np.asmatrix(train_data[:(50000*784)]).reshape(50000, 784)
print("reshape data done")

clf = svm.SVC()
print("start training")
clf.fit(train_data, train_labels[:50000])
print("training done")

# save the model to disk
filename = 'finalized_model_50000_f.sav'
print("save model")
pickle.dump(clf, open(filename, 'wb'))
print("Succeed!")