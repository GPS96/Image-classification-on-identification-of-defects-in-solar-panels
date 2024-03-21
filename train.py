import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import operator

import warnings

warnings.simplefilter("ignore")


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
csv_path = ''
for root, _, files in os.walk('.'):
    for name in files:
        if name == 'data.csv':
            csv_path = os.path.join(root, name)
dataFrame = pd.read_csv(csv_path, sep=';')

train, validation = train_test_split(dataFrame, test_size=0.10, random_state=42)
print(train.shape, validation.shape)
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
# TODO
df_train = ChallengeDataset(train, 'train')
df_val = ChallengeDataset(validation, 'val')
# create an instance of our ResNet model
# TODO

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
# TODO
opti = [True]
Epochs = 200
# Epochs=1
BATCH = [32]
LR = [0.001]
is_dropout = True

for is_adam in opti:
    for Batch_size in BATCH:
        for learning_rate in LR:
            train_DL = t.utils.data.DataLoader(df_train, batch_size=Batch_size, shuffle=True)
            val_DL = t.utils.data.DataLoader(df_val, batch_size=Batch_size)
            print(len(train_DL), len(val_DL))
            resnet = model.ResNet()
            criteria = t.nn.BCELoss()
            if not os.path.isdir('./Adam'):
                os.mkdir('./Adam')
            if not os.path.isdir('./Stochastic'):
                os.mkdir('./Stochastic')

            if is_adam:
                optimizer = t.optim.Adam(resnet.parameters(), lr=learning_rate, weight_decay=0)

            else:
                optimizer = t.optim.SGD(resnet.parameters(), lr=learning_rate, momentum=0.9)


            trained_model = Trainer(model=resnet, crit=criteria, optim=optimizer, train_dl=train_DL, val_test_dl=val_DL,
                                    cuda=True, early_stopping_patience=30)

            # go, go, go... call fit on trainer
            train_loss_list, valid_list = trained_model.fit(epochs=Epochs)
            valid_loss_list = [i[1] for i in valid_list]
            res =train_loss_list, valid_loss_list

            # plot the results
            plt.figure()
            plt.plot(np.arange(len(res[0])), res[0], label='train loss')
            plt.plot(np.arange(len(res[1])), res[1], label='val loss')
            plt.yscale('log')
            plt.legend()
            plt.grid()
            plt.savefig('losses.png')
            plt.show()
            plt.close()

            valid_list.sort(key=operator.itemgetter(2), reverse=True)
            epoch_count, valid_loss, f1_score = valid_list[0]
            f = open("best_results.txt", "w+")
            f.write(f'F1 score on the best model is {f1_score} with valid_loss = {valid_loss} and'
                    f' epoch of {epoch_count} \n Batch size = {Batch_size} and learning rate = {learning_rate}')
            f.close()
            trained_model.save_onnx('best_checkpoint_{:03d}.onnx'.format(epoch_count))
            trained_model.save_onnx('latest_model.onnx')