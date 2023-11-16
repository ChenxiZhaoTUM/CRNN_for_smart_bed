import numpy as np
import torch
from torch.utils.data import Dataset
import realtime_display_CPU as rd_cpu
import unittest

pressureNormalization = True
inputNormalization = True


# slice into time_step

def loader_normalizer(data):
    if pressureNormalization:
        target_data_values = [value['target_data'] for value in data.common_data.values() if
                              len(value['target_data']) > 0]

        if len(target_data_values) > 0:
            data.target_min = np.amin(target_data_values)
            data.target_max = np.amax(target_data_values)

            for value in data.common_data.values():
                target_data = value['target_data']
                normalized_target_data = (target_data - data.target_min) / (data.target_max - data.target_min)
                value['target_data'] = normalized_target_data

            print("Max Pressure:" + str(data.target_max))
            print("Min Pressure:" + str(data.target_min))

    if inputNormalization:
        input_data_values = [value['input_data'] for value in data.common_data.values() if
                             len(value['input_data']) > 0]

        if len(input_data_values) > 0:
            for value in data.common_data.values():
                input_data = value['input_data']
                normalized_input_data = input_data / 4096
                value['input_data'] = normalized_input_data

    return data


class PressureDataset(Dataset):
    TRAIN = 0
    TEST = 2

    def __init__(self, mode=TRAIN, dataDir="./dataset/for_train/", dataDirTest="./dataset/for_test/", time_step=10):
        global pressureNormalization, inputNormalization

        if not (mode == self.TRAIN or mode == self.TEST):
            print("Error - PressureDataset invalid mode " + format(mode))
            exit(1)

        self.mode = mode
        self.dataDir = dataDir
        self.dataDirTest = dataDirTest  # only for mode==self.TEST
        self.common_data = rd_cpu.save_data_from_files(isTest=False)
        self = loader_normalizer(self)
        self.totalLength = len(self.common_data)
        self.time_step = time_step

        if not self.mode == self.TEST:
            # split for train/validation sets (80/20)
            # targetLength = int(0.8 * self.totalLength)
            targetLength = self.totalLength

            self.inputs = []
            self.inputs_sleep = []
            self.targets = []
            self.valiInputs = []
            self.valiInputs_sleep = []
            self.valiTargets = []

            for common_data_id in range(self.totalLength):
                value = self.common_data[common_data_id]
                input_data = value['input_data']
                input_sleep_data = value['input_sleep_data']
                target_data = value['target_data']
                if common_data_id < targetLength:
                    self.inputs.append(input_data)
                    self.inputs_sleep.append(input_sleep_data)
                    self.targets.append(target_data)
                else:
                    self.valiInputs.append(input_data)
                    self.valiInputs_sleep.append(input_sleep_data)
                    self.valiTargets.append(target_data)

            self.valiLength = self.totalLength - targetLength
            self.totalLength = targetLength

        else:
            # test mode
            self.inputs = []
            self.targets = []
            for common_data_id in range(self.totalLength):
                value = self.common_data[common_data_id]
                input_data = value['input_data']
                input_sleep_data = value['input_sleep_data']
                target_data = value['target_data']
                self.inputs.append(input_data)
                self.inputs_sleep.append(input_sleep_data)
                self.targets.append(target_data)

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        input_data_group = self.inputs[idx: (idx + self.time_step)]  # index-th sets data in a batch (a full time step)
        input_sleep_data_group = self.inputs_sleep[idx: (idx + self.time_step)]
        target_data = self.targets[idx + self.time_step]  # only need the last one

        input_data_in_a_full_time_step = torch.zeros(self.time_step, 12, 32, 64)

        for i in range(self.time_step):
            for j in range(11):
                input_data_in_a_full_time_step[i, j, :, :] = torch.tensor(input_sleep_data_group[i][j],
                                                                          dtype=torch.float32)  # Stack along the first dimension

            for j in range(16):
                input_data_in_a_full_time_step[i, 11, :, j * 4: (j + 1) * 4] = torch.tensor(input_data_group[i][j],
                                                                                            dtype=torch.float32)

        target_data_of_the_last_time = torch.from_numpy(target_data).unsqueeze(0)  # to Tensor

        return input_data_in_a_full_time_step, target_data_of_the_last_time

    def denormalize(self, np_array):
        denormalized_data = np_array * (self.target_max - self.target_min) + self.target_min

        return denormalized_data


class ValiDataset(PressureDataset):
    def __init__(self, dataset):
        self.inputs = dataset.valiInputs
        self.inputs_sleep = dataset.valiInputs_sleep
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength

    def __len__(self):
        return self.totalLength

    def __getitem__(self, idx):
        # idx must be smaller than (self.totalLength - self.time_step)

        input_data_group = self.inputs[idx: (idx + self.time_step)]  # index-th sets data in a batch (a full time step)
        input_sleep_data_group = self.inputs_sleep[idx: (idx + self.time_step)]
        target_data = self.targets[idx + self.time_step]  # only need the last one

        input_data_in_a_full_time_step = torch.zeros(self.time_step, 12, 32, 64)

        for i in range(self.time_step):
            for j in range(11):
                input_data_in_a_full_time_step[i, j, :, :] = torch.tensor(input_sleep_data_group[i][j],
                                                                          dtype=torch.float32)  # Stack along the first dimension

            for j in range(16):
                input_data_in_a_full_time_step[i, 11, :, j * 4: (j + 1) * 4] = torch.tensor(input_data_group[i][j],
                                                                                            dtype=torch.float32)

        target_data_of_the_last_time = torch.from_numpy(target_data).unsqueeze(0)  # to Tensor

        return input_data_in_a_full_time_step, target_data_of_the_last_time


class TestPressureDataset(unittest.TestCase):
    def setUp(self):
        self.pressure_dataset = PressureDataset()

    def test_getitem_shape(self):
        idx = 0
        input_data_in_a_full_time_step, target_data_of_the_last_time = self.pressure_dataset[idx]
        self.assertEqual(input_data_in_a_full_time_step.shape, torch.Size([10, 12, 32, 64]))
        self.assertEqual(target_data_of_the_last_time.shape, torch.Size([1, 32, 64]))


if __name__ == '__main__':
    unittest.main()
