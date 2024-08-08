import torch
import torch.nn as nn

# conv2x2 initial version(ours)
class conv2x2_initial_release(nn.Module):
    def __init__(self, c1, c2):
        super(conv2x2_initial_release, self).__init__()
        self.conv = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=2, stride=1, padding=0)
    def forward(self,x):
        x1 = nn.functional.pad(x, (1, 0, 1, 0), mode='constant', value=0)
        x2 = nn.functional.pad(x, (0, 1, 1, 0), mode='constant', value=0)
        x3 = nn.functional.pad(x, (1, 0, 0, 1), mode='constant', value=0)
        x4 = nn.functional.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        x = self.conv(x1)+self.conv(x2)+self.conv(x3)+self.conv(x4)
        return x

# conv2x2 final version(ours)
class conv2x2(nn.Module):
    def __init__(self, c1, c2):
        super(conv2x2, self).__init__()
        self.conv = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=2, stride=1, padding=1)
    def forward(self,x):
        x = self.conv(x)
        x = x[:, :, 1:, 1:]+x[:, :, :-1, :-1]+x[:, :, 1:, :-1]+x[:, :, :-1, 1:]
        return x

# km2x2
class km2x2(nn.Module):
    def __init__(self, c1, c2):
        super(km2x2, self).__init__()
        c = c1*c2//(c1+c2)
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=c, kernel_size=2, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c2, kernel_size=2, stride=1, padding=1)
    def forward(self,x):
        x = self.conv2(self.conv1(x))
        return x

# sp2x2
class sp2x2(nn.Module):
    def __init__(self, c1, c2):
        super(sp2x2, self).__init__()
        self.conv = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=2, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x1 = x[:, :c//4, :, :]
        x2 = x[:, c//4:c//2, :, :]
        x3 = x[:, c//2:c//4*3, :, :]
        x4 = x[:, c//4*3:c, :, :]
        x1 = nn.functional.pad(x1, (1, 0, 1, 0), mode="constant", value=0)  # left top
        x2 = nn.functional.pad(x2, (0, 1, 1, 0), mode="constant", value=0)  # right top
        x3 = nn.functional.pad(x3, (1, 0, 0, 1), mode="constant", value=0)  # left bottom
        x4 = nn.functional.pad(x4, (0, 1, 0, 1), mode="constant", value=0)  # right bottom
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)