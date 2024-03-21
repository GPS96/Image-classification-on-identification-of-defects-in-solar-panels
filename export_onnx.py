import torch as t
import sys
import torchvision as tv
from trainer import Trainer
import model

epoch = int(sys.argv[1])
print(epoch)
#TODO: Enter your model here
model= model.ResNet()

if t.cuda.is_available():
    model.cuda()

crit = t.nn.BCELoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))

