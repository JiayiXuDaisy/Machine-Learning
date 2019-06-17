## How to use
### Install PyTorch
For Linux:
```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

### Install AdverTorch
```
pip install advertorch
```

### Install other packages
Import packages in python files.

## Files
### Python files
**adversarial_defense.py** should run first.

* **adversarial_defense.py** is to perform attack and defense, and generate data files.

* **accuracy.py** is an accuracy test on all three kinds of data..

* **comparison.py** is a visulization of prediction and comparison results.

#### Dataset
Download **fer2013** from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) and add **fer2013.csv** file to **data** folder.

#### Generate other data
Run **adversarial_defense.py** and generate three kinds of data.
```
python adversarial_defense.py
```
Then you can run other .py files.