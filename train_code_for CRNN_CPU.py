import sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from CnnEncoder_RNN_CnnDecoder import weights_init, CRNN
import data_preprocessing_for_CRNN as dp
import utils

##### basic settings #####
# number of training iterations
iterations = 10000
# batch size
batch_size = 50
# time step
time_step = 10
# learning rate
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 3
# data set config
prop = None  # by default, use all from "./dataset/for_train"
# save txt files with per epoch loss?
saveL1 = True
# add Dropout2d layer?
dropout = 0

prefix = ""
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

doLoad = ""  # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))

seed = random.randint(0, 2 ** 32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

##### create pytorch data object with dataset #####
data = dp.PressureDataset()
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
dataValidation = dp.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))

##### setup training #####
epochs = int(iterations / len(trainLoader) + 0.5)
netG = CRNN(channelExponent=expo, dropout=dropout)
print(netG)  # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized ConvNet with {} trainable params ".format(params))
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

    netG.train()
    L1_accum = 0.0
    for batch_index in range(0, len(trainLoader) - time_step):
        print(batch_index)
        inputs_cpu, targets_cpu = trainLoader.dataset[batch_index]
        # torch.Size([50, 10, 12, 32, 64]) and torch.Size([50, 1, 32, 64])

        # print(inputs_cpu.size())
        # print(targets_cpu.size())

        # inputs.data.copy_(inputs_cpu.float())
        # targets.data.copy_(targets_cpu.float())
        #
        # # compute LR decay
        # if decayLr:
        #     currLr = utils.computeLR(epoch, epochs, lrG * 0.1, lrG)
        #     if currLr < lrG:
        #         for g in optimizerG.param_groups:
        #             g['lr'] = currLr
        #
        # netG.zero_grad()
        # gen_out = netG(inputs)
        # gen_out_cpu = gen_out.data.cpu().numpy()
        #
        # lossL1 = criterionL1(gen_out, targets)
        # lossL1.backward()
        #
        # optimizerG.step()
        #
        # lossL1viz = lossL1.item()
        # L1_accum += lossL1viz



    pass
