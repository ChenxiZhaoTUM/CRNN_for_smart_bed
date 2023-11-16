import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from CnnEncoder_RNN_CnnDecoder import CRNN
import data_preprocessing_for_CRNN as dp
import utils

prefix = "CRNN_01_"
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))

suffix = ""
lf = "./" + prefix + "testout{}.txt".format(suffix)
utils.makeDirs(["TEST_CRNN"])

time_step = 10
expo = 3
batch_size = 1

dataset = dp.PressureDataset(mode=dp.PressureDataset.TEST, time_step=time_step)
testLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print("Test batches: {}".format(len(testLoader)))

inputs = Variable(torch.FloatTensor(batch_size, time_step, 12, 32, 64))
targets = Variable(torch.FloatTensor(batch_size, 1, 32, 64))

targets_dn = Variable(torch.FloatTensor(batch_size, 1, 32, 64))
outputs_dn = Variable(torch.FloatTensor(batch_size, 1, 32, 64))

netG = CRNN(channelExponent=expo)
print(netG)  # print full net

doLoad = "CRNN_01_model"
if len(doLoad) > 0:
    netG.load_state_dict(torch.load(doLoad))
    print("Loaded model " + doLoad)

criterionL1 = nn.L1Loss()
criterionL1.cuda()

L1val_accum = 0.0
L1val_dn_accum = 0.0
lossPer_p_accum = 0

# TEST
netG.eval()

for i, testdata in enumerate(testLoader, 0):
    inputs_cpu, targets_cpu = testdata

    inputs.data.copy_(inputs_cpu.float())
    targets.data.copy_(targets_cpu.float())
    outputs = netG(inputs)
    targets_cpu = targets.data.cpu().numpy()
    outputs_cpu = outputs.data.cpu().numpy()

    lossL1 = criterionL1(outputs, targets)
    L1val_accum += lossL1.item()

    lossPer_p = np.sum(np.abs(outputs_cpu - targets_cpu)) / np.sum(np.abs(targets_cpu))
    lossPer_p_accum += lossPer_p.item()

    utils.log(lf, "Test sample %d" % i)
    utils.log(lf, "    pressure:  abs. difference, ratio: %f , %f " % (
        np.sum(np.abs(outputs_cpu - targets_cpu)), lossPer_p.item()))

    targets_denormalized = dataset.denormalize(targets_cpu)
    outputs_denormalized = dataset.denormalize(outputs_cpu)

    targets_denormalized_compare = torch.from_numpy(targets_denormalized)
    outputs_denormalized_compare = torch.from_numpy(outputs_denormalized)

    targets_dn.data.resize_as_(targets_denormalized_compare).copy_(targets_denormalized_compare)
    outputs_dn.data.resize_as_(outputs_denormalized_compare).copy_(outputs_denormalized_compare)

    lossL1_dn = criterionL1(outputs_dn, targets_dn)
    L1val_dn_accum += lossL1_dn.item()

    os.chdir("./TEST_CRNN/")
    utils.imageOut("%04d" % i, inputs[0], targets_denormalized[0], outputs_denormalized[0])
    os.chdir("../")

utils.log(lf, "\n")
L1val_accum /= len(testLoader)
lossPer_p_accum /= len(testLoader)
L1val_dn_accum /= len(testLoader)
utils.log(lf, "Loss percentage of p: %f %% " % (lossPer_p_accum * 100))
utils.log(lf, "L1 error: %f" % L1val_accum)
utils.log(lf, "Denormalized error: %f" % L1val_dn_accum)
utils.log(lf, "\n")

