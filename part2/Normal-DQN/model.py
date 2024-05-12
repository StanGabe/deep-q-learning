import torch
import torch.nn as nn
import torch.nn.functional as F

from config import input_height, input_width, lstm_sequence_length

class Model(nn.Module):
    def __init__(self, action_count):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(3136, 512)
        self.output_layer = nn.Linear(512, action_count)

    def forward(self, input_tensor):
        x = F.relu(self.bn1(self.conv1(input_tensor)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        return self.output_layer(x)
