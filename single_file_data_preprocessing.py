import numpy as np
import torch
from torch.utils.data import Dataset
import realtime_display_CPU as rd_cpu
import unittest
import os

pressureNormalization = True
inputNormalization = True


# change input_data to [12, 32, 64]
# change output_data to [1, 32, 64]
def change_dimension(input_data, input_sleep_data, target_data):
    new_input_data = torch.zeros(12, 32, 64)

    for ch in range(11):
        new_input_data[ch, :, :] = input_sleep_data[ch]  # input_sleep_data to 11 channels

    for j in range(16):
        new_input_data[11, :, j * 4: (j + 1) * 4] = input_data[j]

    new_target_data = torch.from_numpy(target_data).unsqueeze(0)
    return new_input_data, new_target_data


def loader_normalizer(inputs_pressure_arr, targets_arr):
    inputs_pressure_arr_norm = []
    targets_arr_norm = []

    if pressureNormalization:
        if len(targets_arr) > 0:
            # target_min = np.amin(targets_arr)
            # target_max = np.amax(targets_arr)
            target_min = 0
            target_max = 60

            for target_value in targets_arr:
                normalized_target_value = (target_value - target_min) / (target_max - target_min)
                targets_arr_norm.append(normalized_target_value)

            targets_arr = np.array(targets_arr_norm)

    if inputNormalization:
        if len(inputs_pressure_arr) > 0:
            for input_value in inputs_pressure_arr:
                normalized_input_data = input_value / 4096
                inputs_pressure_arr_norm.append(normalized_input_data)
            inputs_pressure_arr = np.array(inputs_pressure_arr_norm)

    return inputs_pressure_arr, targets_arr


##### save common data from file by file #####
def load_data_from_file(csv_file_path):
    filename_without_extension = os.path.splitext(os.path.basename(csv_file_path))[0]
    print(f"The operating file: {filename_without_extension}")

    txt_file_path = os.path.join(os.path.dirname(csv_file_path), f"{filename_without_extension}.txt")
    if not os.path.isfile(txt_file_path):
        print(f"Error: {filename_without_extension}.txt does not exist.")
        exit(1)

    sleep_txt_file_path = os.path.join(os.path.dirname(csv_file_path), f"{filename_without_extension}Sleep.txt")
    if not os.path.isfile(sleep_txt_file_path):
        print(f"Error: {filename_without_extension}Sleep.txt does not exist.")
        exit(1)

    time_arr_txt, value_arr_txt = rd_cpu.deal_with_txt_file(txt_file_path)
    time_arr_sleep_txt, value_arr_sleep_txt = rd_cpu.deal_with_sleep_txt_file(sleep_txt_file_path)
    time_arr_csv, value_arr_csv = rd_cpu.deal_with_csv_file(csv_file_path)

    # do time average
    avg_time_arr_txt, avg_value_arr_txt = rd_cpu.average_by_sec(time_arr_txt, value_arr_txt)
    avg_time_arr_sleep_txt, avg_value_arr_sleep_txt = rd_cpu.average_by_sec(time_arr_sleep_txt, value_arr_sleep_txt)
    avg_time_arr_csv, avg_value_arr_csv = rd_cpu.average_by_sec(time_arr_csv, value_arr_csv)
    avg_value_arr_txt, avg_value_arr_csv = loader_normalizer(avg_value_arr_txt, avg_value_arr_csv)

    avg_value_arr_csv = rd_cpu.reshape_output_value(avg_value_arr_csv)

    ##### save common data from single file in dictionary #####
    inputs_in_single_file = {}
    inputs_sleep_in_single_file = {}
    targets_in_single_file = {}

    for i in range(len(avg_time_arr_txt)):
        time = avg_time_arr_txt[i]
        inputs_in_single_file[time] = avg_value_arr_txt[i]

    for i in range(len(avg_time_arr_sleep_txt)):
        time = avg_time_arr_sleep_txt[i]
        inputs_sleep_in_single_file[time] = avg_value_arr_sleep_txt[i]

    for i in range(len(avg_time_arr_csv)):
        time = avg_time_arr_csv[i]
        targets_in_single_file[time] = avg_value_arr_csv[i]

    common_data_in_single_file = {}
    for time, input_data in inputs_in_single_file.items():
        if time in inputs_sleep_in_single_file and time in targets_in_single_file:
            input_sleep_data = inputs_sleep_in_single_file[time]
            target_data = targets_in_single_file[time]

            new_input_data, new_target_data = change_dimension(input_data, input_sleep_data, target_data)

            common_data_id = len(common_data_in_single_file)
            common_data_in_single_file[common_data_id] = {
                'time': time,
                'input_data': new_input_data,
                'target_data': new_target_data
            }

    print(f"Number of data loaded of {filename_without_extension}:", len(common_data_in_single_file))

    return common_data_in_single_file


def extract_data(common_data_in_single_file, time_step):
    # divide common_data dictionary into inputs and targets array
    data_length_in_single_file = len(common_data_in_single_file)
    inputs = []
    targets = []

    for common_data_id in range(data_length_in_single_file):
        value = common_data_in_single_file[common_data_id]
        input_data = value['input_data']
        target_data = value['target_data']
        inputs.append(input_data)
        targets.append(target_data)

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    # change inputs_data to [10, 12, 32, 64]
    # target_data is still [1, 32, 64]
    X = []
    y = []
    for i in range(data_length_in_single_file - time_step):
        X.append([a for a in inputs[i: (i + time_step)].numpy()])
        y.append(targets[i + time_step].numpy())

    X = np.array(X)
    y = np.array(y)
    return X, y


class PressureDataset(Dataset):
    def __init__(self, csv_file_path, time_step=10):
        self.csv_file_path = csv_file_path
        self.time_step = time_step
        self.common_data_in_single_file = load_data_from_file(self.csv_file_path)
        self.inputs_groups, self.target_groups = extract_data(self.common_data_in_single_file, self.time_step)
        self.total_length = len(self.inputs_groups)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        return self.inputs_groups[idx], self.target_groups[idx]

    def denormalize(self, np_array):
        target_max = 60
        target_min = 0
        denormalized_data = np_array * (target_max - target_min) + target_min

        return denormalized_data


class TestPressureDataset(unittest.TestCase):
    def setUp(self):
        self.pressure_dataset = PressureDataset("./dataset/for_train/duchengsheng.CSV")

    def test_getitem_shape(self):
        data_idx = 0  # file index
        # Number of data loaded of duchengsheng: 554
        inputs_group, target_group = self.pressure_dataset[data_idx]
        self.assertEqual(inputs_group.shape, torch.Size([10, 12, 32, 64]))
        self.assertEqual(target_group.shape, torch.Size([1, 32, 64]))


if __name__ == '__main__':
    unittest.main()
