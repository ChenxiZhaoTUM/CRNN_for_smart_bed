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

    def __init__(self, mode=TRAIN, dataDir="/home/yyc/chenxi/CRNN_for_smart_bed/dataset/for_train", dataDirTest="/home/yyc/chenxi/CRNN_for_smart_bed/dataset/for_test/", time_step=10):
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
            targetLength = int(0.8 * self.totalLength)
            # targetLength = self.totalLength

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
            self.inputs_sleep = []
            self.targets = []
            for common_data_id in range(self.totalLength):
                value = self.common_data[common_data_id]
                input_data = value['input_data']
                input_sleep_data = value['input_sleep_data']
                target_data = value['target_data']
                self.inputs.append(input_data)
                self.inputs_sleep.append(input_sleep_data)
                self.targets.append(target_data)

        self.new_inputs, self.new_targets = self.change_dimension()
        # test code
        print()
        print("The size of train data: ")
        print(self.new_inputs.shape)  # (554, 12, 32, 64)
        print(self.new_targets.shape)  # (554, 1, 32, 64)

        self.inputs_group, self.target_group = self.extract_data()
        # test code
        print()
        print("The size of train data splited by time step: ")
        print(self.inputs_group.shape)  # (544, 10, 12, 32, 64)
        print(self.target_group.shape)  # (544, 1, 32, 64)
        print()

    def __len__(self):
        if self.totalLength > self.time_step:
            length = self.totalLength - self.time_step
        else:
            length = 0
        return length

    def change_dimension(self):
        new_inputs = []
        new_targets = []
        for i in range(self.totalLength):
            input_data = self.inputs[i]
            input_sleep_data = self.inputs_sleep[i]
            target_data = self.targets[i]

            new_input_data = torch.zeros(12, 32, 64)

            for j in range(11):
                new_input_data[j, :, :] = input_sleep_data[j]  # input_sleep_data to 11 channels

            for j in range(16):
                new_input_data[11, :, j * 4: (j + 1) * 4] = input_data[j]

            new_target_data = torch.from_numpy(target_data).unsqueeze(0)

            new_inputs.append(new_input_data)
            new_targets.append(new_target_data)

        new_inputs = np.array(new_inputs)
        new_targets = np.array(new_targets)
        return new_inputs, new_targets

    def extract_data(self):
        X = []
        y = []
        for i in range(self.totalLength - self.time_step):
            X.append([a for a in self.new_inputs[i: (i + self.time_step)]])
            y.append(self.new_targets[i + self.time_step])

        X = np.array(X)
        y = np.array(y)
        return X, y

    def __getitem__(self, idx):
        return self.inputs_group[idx], self.target_group[idx]

    def denormalize(self, np_array):
        denormalized_data = np_array * (self.target_max - self.target_min) + self.target_min

        return denormalized_data


class ValiDataset(PressureDataset):
    def __init__(self, dataset):
        self.inputs = dataset.valiInputs
        self.inputs_sleep = dataset.valiInputs_sleep
        self.targets = dataset.valiTargets
        self.totalLength = dataset.valiLength
        self.time_step = dataset.time_step
        self.inputs_group = dataset.inputs_group
        self.target_group = dataset.target_group

    def __len__(self):
        if self.totalLength > self.time_step:
            length = self.totalLength - self.time_step
        else:
            length = 0
        return length

    def __getitem__(self, idx):
        return self.inputs_group[idx], self.target_group[idx]


class TestPressureDataset(unittest.TestCase):
    def setUp(self):
        self.pressure_dataset = PressureDataset()

    def test_getitem_shape(self):
        idx = 0
        inputs_group, target_group = self.pressure_dataset[idx]
        self.assertEqual(inputs_group.shape, torch.Size([10, 12, 32, 64]))
        self.assertEqual(target_group.shape, torch.Size([1, 32, 64]))


if __name__ == '__main__':
    unittest.main()
