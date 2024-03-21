import torch as t
from sklearn.metrics import f1_score
import os
import shutil
import warnings

warnings.simplefilter("ignore")

class Trainer:

    def __init__(self,
                 model,  # Model to be trained.
                 crit,  # Loss function
                 optim=None,  # Optimizer
                 train_dl=None,  # Training data set
                 val_test_dl=None,  # Validation (or test) data set
                 cuda=True,  # Whether to use the GPU
                 early_stopping_patience=-1,
                 path='checkpoints'):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience
        self.path= 'checkpoints'

        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
            print("Existing Directory deleted")
        os.mkdir(self.path)
        self.f1score = 0

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

        self.all_loss = []
        self.no_progress_duration = 0

    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()},'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      fn,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                    'output': {0: 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients. By default, PyTorch accumulates (sums up) gradients when backward() is called. This behavior is not required here, so you need to ensure that all the gradients are zero before calling the backward.
        # -propagate through the network
        # -calculate the loss
        # -compute gradient by backward propagation
        # -update weights
        # -return the loss
        # TODO
        self._optim.zero_grad()  # starting gradient=0
        prop = self._model(x)  # propagate network
        l = t.squeeze(y).float()
        loss = self._crit(prop, l)
        loss.backward()  # gradient-backward propagation
        self._optim.step()
        return loss.item()

    def val_test_step(self, x, y):

        # predict
        # propagate through the network and calculate the loss and predictions
        # return the loss and the predictions
        # TODO
        pred = self._model(x)
        l = t.squeeze(y).float()
        loss = self._crit(pred, l)
        return loss.item(), pred

    def train_epoch(self):
        # set training mode
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it
        # TODO
        self._model = self._model.train()
        # iterate through the training set
        loss = 0
        for image, label in self._train_dl:

            if self._cuda:
                image = image.to('cuda')
                label = label.to('cuda')
            else:
                image = image.to('cpu')
                label = label.to('cpu')

            loss = loss + self.train_step(x=image, y=label)

        # calculate the average loss for the epoch and return it
        average_loss = loss / len(self._train_dl)
        return average_loss

    def val_test(self):
        # set eval mode. Some layers have different behaviors during training and testing (for example: Dropout, BatchNorm, etc.). To handle those properly, you'd want to call model.eval()
        # disable gradient computation. Since you don't need to update the weights during testing, gradients aren't required anymore.
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        # TODO
        total_loss = 0
        y_true = None
        y_pred = None
        self._model = self._model.eval()  # eval model

        with t.no_grad():
            for i, (image, labels) in enumerate(self._val_test_dl):
                if self._cuda:
                    img = image.to('cuda')
                    labl = labels.to('cuda')
                else:
                    img = image.to('cpu')
                    labl = labels.to('cpu')

                loss, prediction = self.val_test_step(img, labl)
                total_loss = total_loss + loss

                if i == 0:
                    y_true = labels
                    y_pred = prediction
                else:
                    y_true = t.cat((y_true, labels), dim=0)
                    y_pred = t.cat((y_pred, prediction), dim=0)

            s_pred = t.squeeze(y_pred.cpu().round())
            f1 = f1_score(t.squeeze(y_true.cpu()), s_pred, average='weighted')

            self.f1score= f1
            if self.f1score >= 0.70:
                f = open(self.path + "/results.txt", "a+")
                f.write(
                    f'F1 score on valid_set at epoch{self.cnt_epoch} is {self.f1score},valid_loss=' f'{total_loss / len(self._val_test_dl)}\n')
            return total_loss / len(self._val_test_dl)

    def fit(self, epochs=-1):

        assert self._early_stopping_patience > 0 or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch
        # TODO
        train_loss = []
        val_loss = []
        patience_count = 0
        self.cnt_epoch = 0
        f1mean = []

        while True:

            # stop by epoch number
            # train for a epoch and then calculate the loss and metrics on the validation set
            # append the losses to the respective lists
            # use the save_checkpoint function to save the model (can be restricted to epochs with improvement)
            # check whether early stopping should be performed using the early stopping criterion and stop if so
            # return the losses for both training and validation

            if self.cnt_epoch == epochs:
                break
            self.cnt_epoch += 1
            avg_train_loss = self.train_epoch()

            avg_val_loss = self.val_test()
            f1mean.append(self.f1score)

            print('Epoch:{} F1_Score: {} Validation Loss: {}'.format(self.cnt_epoch, self.f1score, avg_val_loss))

            train_loss.append(avg_train_loss)
            val_loss.append((self.cnt_epoch, avg_val_loss, self.f1score))
            if avg_val_loss > 1.04 * val_loss[len(val_loss) - 2][1]:
                patience_count += 1

            if self.f1score > 0.70:
                self.save_checkpoint(self.cnt_epoch)

            if patience_count >= self._early_stopping_patience or self.cnt_epoch >= epochs:
                return train_loss, val_loss





