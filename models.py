from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch


def size_conv(size, kernel, stride=1, padding=0):
    out = int(((size - kernel + 2 * padding) / stride) + 1)
    return out


def size_max_pool(size, kernel, stride=None, padding=0):
    if stride == None:
        stride = kernel
    out = int(((size - kernel + 2 * padding) / stride) + 1)
    return out


# Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_cifar(size):
    feat = size_conv(size, 3, 1, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 3, 1, 1)
    out = size_max_pool(feat, 2, 2)
    return out


# Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_mnist(size):
    feat = size_conv(size, 5, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 5, 1)
    out = size_max_pool(feat, 2, 2)
    return out


# Parameter Initialization
def init_params(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)


class AllCNN_sequential(nn.Module):
    def __init__(self, n_channels=3, num_classes=10, dropout=False, filters_percentage=1., batch_norm=True):
        super(AllCNN_sequential, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.feature1 = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(inplace=True) if dropout else Identity())
        self.feature2 = nn.Sequential(
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(inplace=True) if dropout else Identity())
        self.feature3 = nn.Sequential(
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8 if n_channels == 3 else 7), 
            Flatten(),
        )
        self.features = nn.Sequential(self.feature1, self.feature2, self.feature3)
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        features = self.feature3(self.feature2(self.feature1(x)))
        output = self.classifier(features)
        return output

    def extract_feature(self, x):
        feat1 = self.feature1(x)  # shape:[64, 96, 16, 16]
        feat2 = self.feature2(feat1)  # shape:[64, 96, 8, 8]
        feat3 = self.feature3(feat2)  # shape:[64, 96]

        out = self.classifier(feat3)
        return [feat1, feat2, feat3], out


# for cifar100
class AllCNNImprovedPlus(nn.Module):
    def __init__(self, num_classes=100):
        super(AllCNNImprovedPlus, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),  # 添加 Batch Normalization
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        return x.view(x.size(0), -1)

    def extract_feature(self, x):
        feature1 = nn.Sequential(*list(self.features.children())[:9])
        feature2 = nn.Sequential(*list(self.features.children())[9:18])
        feature3 = nn.Sequential(*list(self.features.children())[18:])

        feat1 = feature1(x)  # shape:[64, 96, 16, 16]
        feat2 = feature2(feat1)  # shape:[64, 96, 8, 8]
        feat3 = feature3(feat2)  # shape:[64, 96]

        out = self.global_avg_pool(feat3)
        out = out.view(out.size(0), -1)
        return [feat1, feat2, feat3], out



class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class VGGWithFeatureExtraction_cifar(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = model.features  # 使用torchvision的VGG16的features部分
        self.avgpool = model.avgpool    # 使用原VGG16的avgpool
        self.classifier = model.classifier  # 使用原VGG16的classifier部分
        self.classifier[6] = nn.Linear(4096, num_classes)  # 修改输出层为 10 个类别

    def forward(self, x: torch.Tensor):
        """
        定义数据流的前向传播。返回经过全连接层后的最终分类结果。
        :param x: 输入数据
        :return: 最终分类输出
        """
        feat1 = self.features[0:4](x)  # 假设前4层是第一个阶段
        feat2 = self.features[4:9](feat1)  # 假设接下来的5层是第二个阶段
        feat3 = self.features[9:16](feat2)  # 假设接下来的7层是第三个阶段
        feat4 = self.features[16:23](feat3)  # 假设接下来的7层是第四个阶段

        # 最终池化和全连接层
        x = self.avgpool(feat4)
        x = torch.flatten(x, 1).cuda()
        out = self.classifier(x)

        return out

    def extract_feature(self, x: torch.Tensor):
        """
        提取卷积层每个阶段的特征图，并返回经过全连接层后的输出。
        :param x: 输入数据
        :return: 一个包含各个阶段特征图的列表，以及经过全连接层后的输出
        """
        feat1 = self.features[0:4](x)  # 假设前4层是第一个阶段
        feat2 = self.features[4:9](feat1)  # 假设接下来的5层是第二个阶段
        feat3 = self.features[9:16](feat2)  # 假设接下来的7层是第三个阶段
        feat4 = self.features[16:23](feat3)  # 假设接下来的7层是第四个阶段

        # 最终池化和全连接层
        x = self.avgpool(feat4)
        x = torch.flatten(x, 1)
        out = self.classifier(x)

        # 返回各阶段的特征和最终分类结果
        return [feat1, feat2, feat3, feat4], out


class VGGWithFeatureExtraction_mnist(nn.Module):
    def __init__(self, model: nn.Module, num_classes: int = 10):
        super().__init__()
        # 修改输入通道为 1（适配灰度图像）
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # 替换第一层
            *list(model.features.children())[1:],  # 继承后续层
        )
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))  # 修改池化层
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor):
        # 提取特征阶段
        feat1 = self.features[0:4](x)  # 第一阶段
        feat2 = self.features[4:9](feat1)  # 第二阶段
        feat3 = self.features[9:16](feat2)  # 第三阶段
        feat4 = self.features[16:23](feat3)  # 第四阶段

        # 池化和分类
        x = self.avgpool(feat4)
        x = torch.flatten(x, 1)
        out = self.classifier(x)

        return out

    def extract_feature(self, x: torch.Tensor):
        feat1 = self.features[0:4](x)
        feat2 = self.features[4:9](feat1)
        feat3 = self.features[9:16](feat2)
        feat4 = self.features[16:23](feat3)

        x = self.avgpool(feat4)
        x = torch.flatten(x, 1)
        out = self.classifier(x)

        return [feat1, feat2, feat3, feat4], out


class ResNet(nn.Module):

    def __init__(self, block, num_block, n_channels=3, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        # output = self.maxpool(output)

        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

    def extract_feature(self, x):
        x = self.conv1(x)
        # x = self.maxpool(x)

        feat1 = self.conv2_x(x)
        feat2 = self.conv3_x(feat1)
        feat3 = self.conv4_x(feat2)
        feat4 = self.conv5_x(feat3)

        out = self.avg_pool(feat4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return [feat1, feat2, feat3, feat4], out


def resnet18(num_classes=10, n_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], n_channels=n_channels, num_classes=num_classes)


def resnet50(num_classes=100, n_channels=3):
    return ResNet(BottleNeck, [3, 4, 6, 3], n_channels=n_channels, num_classes=num_classes)

