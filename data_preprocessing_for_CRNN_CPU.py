import numpy as np
import torch
from torch.utils.data import Dataset
import realtime_display_CPU as rd_cpu
import unittest

pressureNormalization = True
inputNormalization = True


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

    def __init__(self, mode=TRAIN, dataDir="./dataset/for_train/", dataDirTest="./dataset/for_test/"):
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

        if not self.mode == self.TEST:
            # split for train/validation sets (80/20)
            targetLength = 0.8 * self.totalLength

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
        input_data = self.inputs[idx]  # index-th set data
        input_sleep_data = self.inputs_sleep[idx]
        target_data = self.targets[idx]

        new_input_data = torch.zeros(12, 32, 64)

        for i in range(11):
            new_input_data[i, :, :] = input_sleep_data[i]  # index-th set input_sleep_data to 11 channels

        for i in range(16):
            new_input_data[11, :, i * 4: (i + 1) * 4] = input_data[i]

        return new_input_data, torch.from_numpy(target_data)  # to Tensor

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
        input_data = self.inputs[idx]  # index-th set data
        input_sleep_data = self.inputs_sleep[idx]
        target_data = self.targets[idx]

        new_input_data = torch.zeros(12, 32, 64)

        for i in range(11):
            new_input_data[i, :, :] = input_sleep_data[i]  # index-th set input_sleep_data to 11 channels

        for i in range(16):
            new_input_data[11, :, i * 4: (i + 1) * 4] = input_data[i]

        return new_input_data, torch.from_numpy(target_data)  # to Tensor


class TestPressureDataset(unittest.TestCase):
    def setUp(self):
        self.pressure_dataset = PressureDataset()

    def test_getitem_shape(self):
        idx = 0

        new_input_data, target_data = self.pressure_dataset[idx]

        self.assertEqual(new_input_data.shape, torch.Size([12, 32, 64]))
        self.assertEqual(target_data.shape, torch.Size([32, 64]))


if __name__ == '__main__':
    unittest.main()
