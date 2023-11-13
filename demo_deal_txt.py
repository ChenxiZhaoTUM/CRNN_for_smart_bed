import re
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'dataset/for_train', 'duchengsheng.txt')

with open(file_path, 'r') as file:
    first_line = file.readline().strip()

print(first_line)

time_start = first_line.find("[")
time_end = first_line.find("]")
time_str = first_line[time_start + 1: time_end]
time_str = re.sub(r'[^a-zA-Z0-9:.]', '', time_str)

print(time_str)
# print(first_line[time_end + 1: time_end + 5])
# print(first_line[-2:])

pres_decimal_arr = []

if first_line[time_end + 1: time_end + 5] == "AA23" and first_line[-2:] == "55":
    pres_hex_values = first_line[time_end + 5: -2]
    print("Pressure hex values:", pres_hex_values)
    # B50C4F0FFF0F5B0CFF0FFF0F730FC80CF9063707670BAF05450AD409CF06E70E

    if len(pres_hex_values) == 64:
        processed_hex_values = ''.join(
            [pres_hex_values[i + 2: i + 4] + pres_hex_values[i: i + 2] for i in
             range(0, len(pres_hex_values), 4)])

        print("Processed hex values:", processed_hex_values)
        # 0CB50F4F0FFF0C5B0FFF0FFF0F730CC806F907370B6705AF0A4509D406CF0EE7

        pres_decimal_arr = [4095 - int(processed_hex_values[i:i + 4], 16) for i in
                            range(0, len(processed_hex_values), 4)]

print(pres_decimal_arr)
# [3253, 3919, 4095, 3163, 4095, 4095, 3955, 3272, 1785, 1847, 2919, 1455, 2629, 2516, 1743, 3815]
# [842, 176, 0, 932, 0, 0, 140, 823, 2310, 2248, 1176, 2640, 1466, 1579, 2352, 280]
normalized_arr = [x / 4096.0 for x in pres_decimal_arr]
print(normalized_arr)

##### Convert data #####
file_path_convert = os.path.join(current_dir, 'dataset/for_train', 'duchengshengConver.txt')

with open(file_path_convert, 'r') as file:
    first_line_convert = file.readline().strip()

print(first_line_convert)

time_start_convert = first_line_convert.find("[")
time_end_convert = first_line_convert.find("]")

convert_data_arr = [1024-int(x) for x in first_line_convert[time_end_convert + 1:].split(',')]
print(convert_data_arr)
# [41, 131, 62, 95, 10, 26, 9, 42, 115, 368, 547, 709, 229, 253, 537, 369]
normalized_convert_arr = [x / 1024.0 for x in convert_data_arr]
print(normalized_convert_arr)
