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
import DataPreprocessing_SingleFile as dp
from UNet3D import weights_init, UNet3D
import utils

##### detect devices and load files #####
use_cuda = torch.cuda.is_available()  # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
dataDir = "/home/yyc/chenxi/CRNN_for_smart_bed/dataset/for_train"
csv_files = [os.path.join(dataDir, file) for file in os.listdir(dataDir) if file.endswith('.CSV')]

##### basic settings #####
# number of training iterations
iterations = 10000000
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
# save txt files with per epoch loss?
saveL1 = True
# add Dropout2d layer?
dropout = 0

prefix = "UNet3D_01_"
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

doLoad = ""  # optional, path to pre-trained model

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

##### setup training #####
epochs = iterations
netG = UNet3D(channelExponent=expo, dropout=dropout)
print(netG)  # print full net
model_parameters = filter(lambda p: p.requires_grad, netG.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Initialized UNet3D with {} trainable params ".format(params))
print()

netG.apply(weights_init)
if len(doLoad) > 0:
    netG.load_state_dict(torch.load(doLoad))
    # netG.load_state_dict(torch.load(doLoad, map_location=torch.device('cpu')))
    print("Loaded model " + doLoad)
netG.cuda()

criterionL1 = nn.L1Loss()
optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(0.5, 0.999), weight_decay=0.0)
criterionL1.cuda()

inputs = Variable(torch.FloatTensor(batch_size, time_step, 12, 32, 64))
targets = Variable(torch.FloatTensor(batch_size, 1, 32, 64))
inputs = inputs.cuda()
targets = targets.cuda()

##### training begins #####
for epoch in range(epochs):
    print()
    print("Starting epoch {} / {}".format((epoch + 1), epochs))

    # TRAIN
    netG.train()
    L1_accum = 0.0
    train_times = 0
    vali_times = 0

    random.shuffle(train_files)
    for train_file in train_files:
        train_set = dp.PressureDataset(train_file)
        trainLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        print("Training batches: {}".format(len(trainLoader)))

        for batch_idx, traindata in enumerate(trainLoader, 0):
            inputs_groups, target_groups = traindata

            # test code
            # print(f"batch_idx: {batch_idx}")
            # print(inputs_groups.size())  # torch.Size([50, 10, 12, 32, 64])
            # print(target_groups.size())  # torch.Size([50, 1, 32, 64])

            inputs_cpu = inputs_groups.float().cuda()
            targets_cpu = target_groups.float().cuda()
            inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
            targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

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

            if batch_idx == len(trainLoader) - 1:
                logline = "Epoch: {}, batch-idx: {}, L1: {}".format(epoch, batch_idx, lossL1viz)
                print(logline)

            targets_denormalized = train_set.denormalize(target_groups.cpu().numpy())
            outputs_denormalized = train_set.denormalize(gen_out_cpu)

            if lossL1viz < 0.005:
                for j in range(batch_size):
                    utils.makeDirs(["TRAIN_UNet3D_0.005"])
                    utils.imageOut("TRAIN_UNet3D_0.005/epoch{}_{}_{}".format(epoch, batch_idx, j), inputs_groups[j],
                                   targets_denormalized[j], outputs_denormalized[j])

            if lossL1viz < 0.01:
                torch.save(netG.state_dict(), prefix + "model")

            train_times += 1

    # VALIDATION
    netG.eval()
    L1val_accum = 0.0

    random.shuffle(vali_files)
    for vali_file in vali_files:
        vali_set = dp.PressureDataset(vali_file)
        valiLoader = DataLoader(vali_set, batch_size=batch_size, shuffle=False, drop_last=True)
        print("Validation batches: {}".format(len(valiLoader)))

        for batch_idx, validata in enumerate(valiLoader, 0):
            inputs_groups, target_groups = validata
            inputs_cpu = inputs_groups.float().cuda()
            targets_cpu = target_groups.float().cuda()
            inputs.data.resize_as_(inputs_cpu).copy_(inputs_cpu)
            targets.data.resize_as_(targets_cpu).copy_(targets_cpu)

            outputs = netG(inputs)
            outputs_cpu = outputs.data.cpu().numpy()

            lossL1 = criterionL1(outputs, targets)
            L1val_accum += lossL1.item()

            targets_denormalized = vali_set.denormalize(target_groups.cpu().numpy())
            outputs_denormalized = vali_set.denormalize(outputs_cpu)

            if lossL1viz < 0.005:
                for j in range(batch_size):
                    utils.makeDirs(["VALIDATION_UNet3D_0.005"])
                    utils.imageOut("VALIDATION_UNet3D_0.005/epoch{}_{}_{}".format(epoch, batch_idx, j), inputs_groups[j],
                                   targets_denormalized[j], outputs_denormalized[j])
                    
            vali_times += 1

    L1_accum /= train_times
    L1val_accum /= vali_times
    if saveL1:
        if epoch == 0:
            utils.resetLog(prefix + "L1.txt")
            utils.resetLog(prefix + "L1val.txt")
        utils.log(prefix + "L1.txt", "{} ".format(L1_accum), False)
        utils.log(prefix + "L1val.txt", "{} ".format(L1val_accum), False)
