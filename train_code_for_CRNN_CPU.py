import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import data_preprocessing_for_CRNN as dp
import utils
from CnnEncoder_RNN_CnnDecoder import weights_init, CRNN

##### basic settings #####
# number of training iterations
iterations = 10000000
# batch size
batch_size = 100
# time step
time_step = 10
# learning rate
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 3
# save txt files with per epoch loss?
saveL1 = True
# add Dropout2d layer?
dropout = 0

prefix = "CRNN_01_"
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

doLoad = "CRNN_01_model"  # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))

seed = random.randint(0, 2 ** 32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

##### create pytorch data object with dataset #####
data = dp.PressureDataset(time_step=time_step)
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
dataValidation = dp.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))
print()

##### setup training #####
epochs = int(iterations / len(trainLoader) + 0.5)
netG = CRNN(channelExponent=expo, dropout=dropout)
print(netG)  # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized CRNN with {} trainable params ".format(params))
print()

netG.apply(weights_init)
if len(doLoad) > 0:
    # netG.load_state_dict(torch.load(doLoad))
    netG.load_state_dict(torch.load(doLoad, map_location=torch.device('cpu')))
    print("Loaded model " + doLoad)

criterionL1 = nn.L1Loss()
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)

inputs = Variable(torch.FloatTensor(batch_size, time_step, 12, 32, 64))
targets = Variable(torch.FloatTensor(batch_size, 1, 32, 64))

##### training begins #####
for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch + 1), epochs))

    # TRAIN
    netG.train()
    L1_accum = 0.0
    for i, traindata in enumerate(trainLoader, 0):
        inputs_cpu, targets_cpu = traindata

        # test code
        # print(i)
        # print(inputs_cpu.size())  # torch.Size([50, 10, 12, 32, 64])
        # print(targets_cpu.size())  # torch.Size([50, 1, 32, 64])

        inputs.data.copy_(inputs_cpu.float())
        targets.data.copy_(targets_cpu.float())

        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG * 0.1, lrG)
            if currLr < lrG:
                for g in optimizerG.param_groups:
                    g['lr'] = currLr

        netG.zero_grad()
        gen_out = netG(inputs)
        gen_out_cpu = gen_out.data.cpu().numpy()

        lossL1 = criterionL1(gen_out, targets)
        lossL1.backward()

        optimizerG.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz

        if i == len(trainLoader) - 1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, i, lossL1viz)
            print(logline)

        targets_denormalized = data.denormalize(targets_cpu.cpu().numpy())
        outputs_denormalized = data.denormalize(gen_out_cpu)

        if lossL1viz < 0.002:
            for j in range(batch_size):
                utils.makeDirs(["TRAIN_CRNN_0.002"])
                utils.imageOut("TRAIN_CRNN_0.002/epoch{}_{}_{}".format(epoch, i, j), inputs[j],
                               targets_denormalized[j], outputs_denormalized[j])

        if lossL1viz < 0.004:
            torch.save(netG.state_dict(), prefix + "model")

    # VALIDATION
    netG.eval()
    L1val_accum = 0.0
    for i, validata in enumerate(valiLoader, 0):
        inputs_cpu, targets_cpu = validata

        # test code
        # print(i)
        # print(inputs_cpu.size())  # torch.Size([50, 10, 12, 32, 64])
        # print(targets_cpu.size())  # torch.Size([50, 1, 32, 64])

        inputs.data.copy_(inputs_cpu.float())
        targets.data.copy_(targets_cpu.float())

        outputs = netG(inputs)
        outputs_cpu = outputs.data.cpu().numpy()

        lossL1 = criterionL1(outputs, targets)
        L1val_accum += lossL1.item()

        targets_denormalized = data.denormalize(targets_cpu.cpu().numpy())
        outputs_denormalized = data.denormalize(outputs_cpu)

        if lossL1viz < 0.004:
            for j in range(batch_size):
                utils.makeDirs(["VALIDATION_CRNN_0.004"])
                utils.imageOut("VALIDATION_CRNN_0.004/epoch{}_{}_{}".format(epoch, i, j), inputs[j],
                               targets_denormalized[j], outputs_denormalized[j])

    L1_accum /= len(trainLoader)
    if saveL1:
        if epoch == 0:
            utils.resetLog(prefix + "L1.txt")
            utils.resetLog(prefix + "L1val.txt")
        utils.log(prefix + "L1.txt", "{} ".format(L1_accum), False)
        utils.log(prefix + "L1val.txt", "{} ".format(L1val_accum), False)
