First use `python preprocess_fer2013.py` to preprocess the data

Then use `python mainpro_FER.py --model VGG19 --bs 128 --lr 0.01` to train and evaluate the VGG19.

Then use `python mainpro_FER.py --model Resnet18 --bs 128 --lr 0.01` to train and evaluate the TesNet18.

Then use `python combine.py --bs 128 --lr 0.01` to train and evaluate the our combination model.

