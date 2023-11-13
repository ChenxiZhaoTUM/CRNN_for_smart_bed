import torch
import torch.nn as nn

rnn = nn.LSTM(10, 20, 2)  # input_size, hidden_size, num_layers
input = torch.randn(5, 3, 10)  # batch_size, time_step, input_size
# h0 = torch.randn(2, 3, 20)
# c0 = torch.randn(2, 3, 20)
# output, (hn, cn) = rnn(input, (h0, c0))
output, (hn, cn) = rnn(input)
print(output.shape)  # torch.Size([5, 3, 20])  # batch_size, time_step, output_size
print(hn.shape)  # torch.Size([2, 3, 20])  # num_layers, time_step, output_size
print(cn.shape)  # torch.Size([2, 3, 20])  # num_layers, time_step, output_size
