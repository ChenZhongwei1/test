import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

# x = torch.randn(3)
# x = Variable(x, requires_grad=True)
#
# y = x * x
# print(x)
# print(y)
# y.backward(torch.FloatTensor([1, 0.1, 0.1]))
# print(x.grad)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                   [9.779], [6.182], [7.59], [2.167], [7.042],
                   [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

print("hello")
