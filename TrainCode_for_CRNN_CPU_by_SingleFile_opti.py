import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset, Subset
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
full_train_dataset = ConcatDataset([dp.PressureDataset(train_file) for train_file in train_files])
# train_loaders = [DataLoader(Subset(full_train_dataset, indices=[i]), batch_size=batch_size, shuffle=True, drop_last=True)
#                  for i in range(len(train_files))]
full_vali_dataset = ConcatDataset([dp.PressureDataset(vali_file) for vali_file in vali_files])

##### setup training #####
epochs = iterations
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
    train_times = 0

    random.shuffle(train_files)
    for file_idx, train_file in enumerate(train_files):
        subset = Subset(full_train_dataset, indices=[file_idx])
        print(f"Subset {file_idx} size: {len(subset)}")
        trainLoader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
        print("Training batches: {}".format(len(trainLoader)))

        for batch_idx, traindata in enumerate(trainLoader, 0):
            inputs_groups, target_groups = traindata

            # test code
            # print(f"batch_idx: {batch_idx}")
            # print(inputs_groups.size())  # torch.Size([50, 10, 12, 32, 64])
            # print(target_groups.size())  # torch.Size([50, 1, 32, 64])

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

            if batch_idx == len(trainLoader) - 1:
                logline = "Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, batch_idx, lossL1viz)
                print(logline)

            targets_denormalized = trainLoader.denormalize(target_groups.cpu().numpy())
            outputs_denormalized = trainLoader.denormalize(gen_out_cpu)

            if lossL1viz < 1:
                for j in range(batch_size):
                    utils.makeDirs(["TRAIN_CRNN_1"])
                    utils.imageOut("TRAIN_CRNN_1/epoch{}_{}_{}".format(epoch, batch_idx, j), inputs[j],
                                   targets_denormalized[j], outputs_denormalized[j])

            if lossL1viz < 1:
                torch.save(netG.state_dict(), prefix + "model")

            train_times += 1

    # VALIDATION
    netG.eval()
    L1val_accum = 0.0

    random.shuffle(vali_files)
    for file_idx, vali_file in enumerate(vali_files):
        subset = Subset(full_vali_dataset, indices=[file_idx])
        print(f"Subset {file_idx} size: {len(subset)}")
        valiLoader = DataLoader(subset, batch_size=batch_size, shuffle=False, drop_last=True)
        print("Validation batches: {}".format(len(valiLoader)))

        for batch_idx, validata in enumerate(valiLoader, 0):
            inputs_groups, target_groups = validata
            inputs.data.copy_(inputs_groups.float())
            targets.data.copy_(target_groups.float())

            outputs = netG(inputs)
            outputs_cpu = outputs.data.cpu().numpy()

            lossL1 = criterionL1(outputs, targets)
            L1val_accum += lossL1.item()

            targets_denormalized = valiLoader.denormalize(target_groups.cpu().numpy())
            outputs_denormalized = valiLoader.denormalize(outputs_cpu)

            if lossL1viz < 1:
                for j in range(batch_size):
                    utils.makeDirs(["VALIDATION_CRNN_1"])
                    utils.imageOut("VALIDATION_CRNN_1/epoch{}_{}_{}".format(epoch, batch_idx, j), inputs[j],
                                   targets_denormalized[j], outputs_denormalized[j])

    L1_accum /= train_times
    if saveL1:
        if epoch == 0:
            utils.resetLog(prefix + "L1.txt")
            utils.resetLog(prefix + "L1val.txt")
        utils.log(prefix + "L1.txt", "{} ".format(L1_accum), False)
        utils.log(prefix + "L1val.txt", "{} ".format(L1val_accum), False)
