import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, channel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.shortcut = nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Network(nn.Module):
    def __init__(self, num_feats, hidden_sizes, num_classes, feat_dim=10):
        super(Network, self).__init__()

        self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]

        self.layers = []
        for idx, channel_size in enumerate(hidden_sizes):
            self.layers.append(nn.Conv2d(in_channels=self.hidden_sizes[idx],
                                         out_channels=self.hidden_sizes[idx + 1],
                                         kernel_size=3, stride=2, bias=False))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(BasicBlock(channel_size=channel_size))

        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(self.hidden_sizes[-2], self.hidden_sizes[-1], bias=False)

        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)

    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)

        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])

        label_output = self.linear_label(output)
        label_output = label_output / torch.norm(self.linear_label.weight, dim=1)

        # Create the feature embedding for the Center Loss
        closs_output = self.linear_closs(output)
        closs_output = self.relu_closs(closs_output)

        return closs_output, label_output


class ResnetBlock(nn.Module):
    def __init__(self, channel_size, out_channel, stride = 1 ):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, out_channel, kernel_size=3, stride = stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride= 1, padding =1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Conv2d(channel_size, out_channel, kernel_size=1, stride = stride)

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(self.shortcut(x) + out)
        return out

class BottleNeck(nn.Module):
    def __init__(self, channel_size, out_channel, stride =1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, int(out_channel/4), kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(int(out_channel/4))

        self.conv2 = nn.Conv2d(int(out_channel/4), int(out_channel/4), kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d( int(out_channel/4) )

        self.conv3 = nn.Conv2d(int(out_channel/4), out_channel, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Conv2d(channel_size, out_channel, kernel_size=1, stride = stride)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = F.relu(self.shortcut(x) + out)
        return out

class Resnet(nn.Module):
    def __init__(self, num_features = 3, hidden_layers = [2,2,2], num_classes = 2300):
        super(Resnet, self).__init__()

        # 32x32
        self.conv1 = nn.Conv2d(num_features, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)
        # 16 x 16
        self.layer1 = self.__layer__(64, 256, hidden_layers[0])
        # 8x8
        self.layer2 = self.__layer__(256, 512, hidden_layers[1])
        # 4x4
        self.layer3 = self.__layer__(512, 1024, hidden_layers[2])
        # 2x2
        self.layer4 = self.__layer__(1024, 2048, hidden_layers[3])

        # 1x1
        #self.averagepooling = nn.AvgPool2d((1,1))
        self.linear_label = nn.Linear(2048, num_classes)

    def __layer__(self, channel_size, out_channel, hidden_layers):
        block = []
        for i in range(hidden_layers-1):
            block.append(BottleNeck(channel_size, channel_size, stride=1))
        block.append(BottleNeck(channel_size, out_channel, stride=2))
        return nn.Sequential(*block)

    def forward(self, x, evalMode=False):
        out = F.relu(self.bn1(self.conv1(x)))
        #print("out 1 should be 10*64*32*32",out.size())
        out = F.max_pool2d(out, (3,3), 1, 1)
        #print("out 2 should be 10*64*16*16",out.size())
        out = self.layer1(out)
        #print("out 3 should be 10*128*8*8", out.size())
        out = self.layer2(out)
        #print("out 4 should be 10*256*4*4", out.size())
        out = self.layer3(out)
        #print("out 5 should be 10*512*2*2", out.size())
        out0 = F.avg_pool2d(out,(2,2))
        #print("out 6 should be 10*512*1*1",out.size())
        out1 = torch.flatten(out0, 1)
        #print("out 7 should be 10*512",out.size())
        out2 = self.linear_label(out1)
        #print("out 8 should be 10*2300",out.size())
        return out0 , out2

