import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import single_file_data_preprocessing as dp
from CnnEncoder_RNN_CnnDecoder import weights_init, CRNN
import utils

##### detect devices and load files #####
use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
dataDir = "./dataset/for_train"
csv_files = [os.path.join(dataDir, file) for file in os.listdir(dataDir) if file.endswith('.CSV')]

##### basic settings #####
# number of training iterations
iterations = 10000000
# batch size
batch_size = 1  # notice that here must be 1
# (because number of samples in files are different, DataLoader is forbidden)
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
print()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

##### create pytorch data object with dataset #####
# train, vali split
train_files, vali_files = train_test_split(csv_files, test_size=0.2, random_state=42)
train_set = dp.PressureDataset(train_files)
vali_set = dp.PressureDataset(vali_files)
# data loading params, batch_size represents number of files
trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
valiLoader = DataLoader(vali_set, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
print("Validation batches: {}".format(len(valiLoader)))
print()
""" 
test code
inputs_groups, target_groups = train_set[0]
print(inputs_groups.shape)
print(target_groups.shape)

The operating file: wubing_ry
Number of data loaded of wubing_ry: 602
(592, 10, 12, 32, 64)
(592, 1, 32, 64)
"""

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

##### training begins #####
for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch + 1), epochs))

    # TRAIN
    netG.train()
    L1_accum = 0.0
    for batch_idx, traindata in enumerate(trainLoader, 0):
        print(f"batch_idx: {batch_idx}")
        input_files_in_one_batch, target_files_in_one_batch = traindata  # in one batch (i.e. batch_size)

        for file_idx in range(len(input_files_in_one_batch)):  # only one file, file_idx==0
            inputs_groups = input_files_in_one_batch[file_idx]  # data pairs in one file
            target_groups = target_files_in_one_batch[file_idx]
            # test code
            # print(inputs_groups.size())  # torch.Size([544, 10, 12, 32, 64])
            # print(target_groups.size())  # torch.Size([544, 1, 32, 64])

            inputs = Variable(torch.FloatTensor(inputs_groups.size(0), time_step, 12, 32, 64))
            targets = Variable(torch.FloatTensor(inputs_groups.size(0), 1, 32, 64))
            inputs.data.copy_(inputs_groups.float())
            targets.data.copy_(target_groups.float())

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

            targets_denormalized = train_set.denormalize(inputs_groups.cpu().numpy())
            outputs_denormalized = train_set.denormalize(gen_out_cpu)

            if lossL1viz < 0.002:
                for j in range(batch_size):
                    utils.makeDirs(["TRAIN_CRNN_0.002"])
                    utils.imageOut("TRAIN_CRNN_0.002/epoch{}_{}_{}".format(epoch, i, j), inputs[j],
                                   targets_denormalized[j], outputs_denormalized[j])

            if lossL1viz < 0.004:
                torch.save(netG.state_dict(), prefix + "model")

        if batch_idx == len(trainLoader) - 1:
            logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, batch_idx, lossL1viz)
            print(logline)





