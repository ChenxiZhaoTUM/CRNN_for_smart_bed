import re
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'dataset/for_train', 'duchengshengSleep.txt')

with open(file_path, 'r') as file:
    first_line = file.readline().strip()

print(first_line)

time_start = first_line.find("[")
time_end = first_line.find("]")
time_str = first_line[time_start + 1: time_end]
time_str = re.sub(r'[^a-zA-Z0-9:.]', '', time_str)

print(time_str)
print(first_line[time_end + 1: time_end + 5])
print(first_line[-2:])

if first_line[time_end + 1: time_end + 5] == "AB11" and first_line[-2:] == "55":
    pres_hex_values = first_line[time_end + 5: -2]
    print("Pressure hex values:", pres_hex_values)

    if len(pres_hex_values) == 28:
        processed_hex_values = pres_hex_values[0: 12]

        processed_hex_values = processed_hex_values + ''.join(
            [pres_hex_values[i + 2: i + 4] + pres_hex_values[i: i + 2] for i in
             range(12, 20, 4)])

        processed_hex_values = processed_hex_values + pres_hex_values[20: 22]

        processed_hex_values = processed_hex_values + ''.join(
            [pres_hex_values[24: 26] + pres_hex_values[22: 24]])

        processed_hex_values = processed_hex_values + pres_hex_values[26:]

        print(processed_hex_values)

        pres_decimal_arr = [int(processed_hex_values[i:i + 2], 16) for i in range(0, 12, 2)]
        pres_decimal_arr.extend([int(processed_hex_values[12:16], 16)])
        pres_decimal_arr.extend([int(processed_hex_values[16:20], 16)])
        pres_decimal_arr.append(int(processed_hex_values[20:22], 16))
        pres_decimal_arr.append(int(processed_hex_values[22:26], 16))
        pres_decimal_arr.append(int(processed_hex_values[26:28], 16))
        print(pres_decimal_arr)
