import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
 
# 定义ResNet18/34的残差结构，为2个3x3的卷积
class BasicBlock(nn.Module):
    # 判断残差结构中，主分支的卷积核个数是否发生变化，不变则为1
    expansion = 1
 
    # init()：进行初始化，申明模型中各层的定义
    # downsample=None对应实线残差结构，否则为虚线残差结构
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        # 使用批量归一化
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 使用ReLU作为激活函数
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
 
    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 残差块保留原始输入
        identity = x
        # 如果是虚线残差结构，则进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # -----------------------------------------
        out = self.conv2(out)
        out = self.bn2(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
 
        return out
 
 
# 定义ResNet50/101/152的残差结构，为1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # expansion是指在每个小残差块内，减小尺度增加维度的倍数，如64*4=256
    # Bottleneck层输出通道是输入的4倍
    expansion = 4
 
    # init()：进行初始化，申明模型中各层的定义
    # downsample=None对应实线残差结构，否则为虚线残差结构，专门用来改变x的通道数
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
 
        width = int(out_channel * (width_per_group / 64.)) * groups
 
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        # 使用批量归一化
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        # 使用ReLU作为激活函数
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
     
 
    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # 残差块保留原始输入
        identity = x
        # 如果是虚线残差结构，则进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
 
        return out
 

class ResNet50(nn.Module):
    def __init__(self,num_classes=10,num_channels = 64,include_top=True):
        super(ResNet50,self).__init__()
        self.include_top = include_top
        self.block1_channel = num_channels
        self.block2_channel = num_channels * 2
        self.block3_channel = num_channels * 4
        self.block4_channel = num_channels * 8
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.block1_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.block1_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)


        ##block1 * 3
        self.block1_downsample = nn.Sequential(
                nn.Conv2d(self.block1_channel, self.block1_channel*4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.block1_channel*4))
        self.block1_conv1_1 = nn.Conv2d(in_channels=self.block1_channel, out_channels=self.block1_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block1_bn1_1 = nn.BatchNorm2d(self.block1_channel)
        # -----------------------------------------
        self.block1_conv1_2 = nn.Conv2d(in_channels=self.block1_channel, out_channels=self.block1_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block1_bn1_2 = nn.BatchNorm2d(self.block1_channel)
        # -----------------------------------------
        self.block1_conv1_3 = nn.Conv2d(in_channels=self.block1_channel, out_channels=self.block1_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block1_bn1_3 = nn.BatchNorm2d(self.block1_channel * 4)
        # ------------------------------------------------------------------------------
        self.block1_conv2_1 = nn.Conv2d(in_channels=self.block1_channel*4, out_channels=self.block1_channel,
                               kernel_size=1, stride=1, bias=False)
  
        self.block1_bn2_1 = nn.BatchNorm2d(self.block1_channel)
        # -----------------------------------------
        self.block1_conv2_2 = nn.Conv2d(in_channels=self.block1_channel, out_channels=self.block1_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block1_bn2_2 = nn.BatchNorm2d(self.block1_channel)
        # -----------------------------------------
        self.block1_conv2_3 = nn.Conv2d(in_channels=self.block1_channel, out_channels=self.block1_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block1_bn2_3 = nn.BatchNorm2d(self.block1_channel * 4)
        #--------------------------------------------------------------------------------
        self.block1_conv3_1 = nn.Conv2d(in_channels=self.block1_channel*4, out_channels=self.block1_channel,
                               kernel_size=1, stride=1, bias=False)
   
        self.block1_bn3_1 = nn.BatchNorm2d(self.block1_channel)
        # -----------------------------------------
        self.block1_conv3_2 = nn.Conv2d(in_channels=self.block1_channel, out_channels=self.block1_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block1_bn3_2 = nn.BatchNorm2d(self.block1_channel)
        # -----------------------------------------
        self.block1_conv3_3 = nn.Conv2d(in_channels=self.block1_channel, out_channels=self.block1_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block1_bn3_3 = nn.BatchNorm2d(self.block1_channel * 4)
        
        self.relu = nn.ReLU(inplace=True)


        ###################################block2########################################
        self.block2_downsample = nn.Sequential(
                nn.Conv2d(self.block1_channel*4, self.block2_channel*4, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(self.block2_channel*4))
        self.block2_conv1_1 = nn.Conv2d(in_channels=self.block1_channel*4, out_channels=self.block2_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block2_bn1_1 = nn.BatchNorm2d(self.block2_channel)
        # -----------------------------------------
        self.block2_conv1_2 = nn.Conv2d(in_channels=self.block2_channel, out_channels=self.block2_channel,
                               kernel_size=3, stride=2, bias=False, padding=1)
        self.block2_bn1_2 = nn.BatchNorm2d(self.block2_channel)
        # -----------------------------------------
        self.block2_conv1_3 = nn.Conv2d(in_channels=self.block2_channel, out_channels=self.block2_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block2_bn1_3 = nn.BatchNorm2d(self.block2_channel * 4)
        # ------------------------------------------------------------------------------
        self.block2_conv2_1 = nn.Conv2d(in_channels=self.block2_channel * 4, out_channels=self.block2_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block2_bn2_1 = nn.BatchNorm2d(self.block2_channel)
        # -----------------------------------------
        self.block2_conv2_2 = nn.Conv2d(in_channels=self.block2_channel, out_channels=self.block2_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block2_bn2_2 = nn.BatchNorm2d(self.block2_channel)
        # -----------------------------------------
        self.block2_conv2_3 = nn.Conv2d(in_channels=self.block2_channel, out_channels=self.block2_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block2_bn2_3 = nn.BatchNorm2d(self.block2_channel* 4)
        # ------------------------------------------------------------------------------
        self.block2_conv3_1 = nn.Conv2d(in_channels=self.block2_channel * 4, out_channels=self.block2_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block2_bn3_1 = nn.BatchNorm2d(self.block2_channel)
        # -----------------------------------------
        self.block2_conv3_2 = nn.Conv2d(in_channels=self.block2_channel, out_channels=self.block2_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block2_bn3_2 = nn.BatchNorm2d(self.block2_channel)
        # -----------------------------------------
        self.block2_conv3_3 = nn.Conv2d(in_channels=self.block2_channel, out_channels=self.block2_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block2_bn3_3 = nn.BatchNorm2d(self.block2_channel* 4)
        # ------------------------------------------------------------------------------
        self.block2_conv4_1 = nn.Conv2d(in_channels=self.block2_channel * 4, out_channels=self.block2_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block2_bn4_1 = nn.BatchNorm2d(self.block2_channel)
        # -----------------------------------------
        self.block2_conv4_2 = nn.Conv2d(in_channels=self.block2_channel, out_channels=self.block2_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block2_bn4_2 = nn.BatchNorm2d(self.block2_channel)
        # -----------------------------------------
        self.block2_conv4_3 = nn.Conv2d(in_channels=self.block2_channel, out_channels=self.block2_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block2_bn4_3 = nn.BatchNorm2d(self.block2_channel* 4)
        # ------------------------------------------------------------------------------

        ###################################block3########################################
        self.block3_downsample = nn.Sequential(
                nn.Conv2d(self.block2_channel*4, self.block3_channel*4, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(self.block3_channel*4))
        self.block3_conv1_1 = nn.Conv2d(in_channels=self.block2_channel*4, out_channels=self.block3_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block3_bn1_1 = nn.BatchNorm2d(self.block3_channel)
        # -----------------------------------------
        self.block3_conv1_2 = nn.Conv2d(in_channels=self.block3_channel, out_channels=self.block3_channel,
                               kernel_size=3, stride=2, bias=False, padding=1)
        self.block3_bn1_2 = nn.BatchNorm2d(self.block3_channel)
        # -----------------------------------------
        self.block3_conv1_3 = nn.Conv2d(in_channels=self.block3_channel, out_channels=self.block3_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block3_bn1_3 = nn.BatchNorm2d(self.block3_channel * 4)
        # ------------------------------------------------------------------------------
        self.block3_conv2_1 = nn.Conv2d(in_channels=self.block3_channel* 4, out_channels=self.block3_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block3_bn2_1 = nn.BatchNorm2d(self.block3_channel)
        # -----------------------------------------
        self.block3_conv2_2 = nn.Conv2d(in_channels=self.block3_channel, out_channels=self.block3_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block3_bn2_2 = nn.BatchNorm2d(self.block3_channel)
        # -----------------------------------------
        self.block3_conv2_3 = nn.Conv2d(in_channels=self.block3_channel, out_channels=self.block3_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block3_bn2_3 = nn.BatchNorm2d(self.block3_channel* 4)
        # ------------------------------------------------------------------------------
        self.block3_conv3_1 = nn.Conv2d(in_channels=self.block3_channel * 4, out_channels=self.block3_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block3_bn3_1 = nn.BatchNorm2d(self.block3_channel)
        # -----------------------------------------
        self.block3_conv3_2 = nn.Conv2d(in_channels=self.block3_channel, out_channels=self.block3_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block3_bn3_2 = nn.BatchNorm2d(self.block3_channel)
        # -----------------------------------------
        self.block3_conv3_3 = nn.Conv2d(in_channels=self.block3_channel, out_channels=self.block3_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block3_bn3_3 = nn.BatchNorm2d(self.block3_channel* 4)
        # ------------------------------------------------------------------------------
        self.block3_conv4_1 = nn.Conv2d(in_channels=self.block3_channel * 4, out_channels=self.block3_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block3_bn4_1 = nn.BatchNorm2d(self.block3_channel)
        # -----------------------------------------
        self.block3_conv4_2 = nn.Conv2d(in_channels=self.block3_channel, out_channels=self.block3_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block3_bn4_2 = nn.BatchNorm2d(self.block3_channel)
        # -----------------------------------------
        self.block3_conv4_3 = nn.Conv2d(in_channels=self.block3_channel, out_channels=self.block3_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block3_bn4_3 = nn.BatchNorm2d(self.block3_channel* 4)
        # ------------------------------------------------------------------------------
        self.block3_conv5_1 = nn.Conv2d(in_channels=self.block3_channel * 4, out_channels=self.block3_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block3_bn5_1 = nn.BatchNorm2d(self.block3_channel)
        # -----------------------------------------
        self.block3_conv5_2 = nn.Conv2d(in_channels=self.block3_channel, out_channels=self.block3_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block3_bn5_2 = nn.BatchNorm2d(self.block3_channel)
        # -----------------------------------------
        self.block3_conv5_3 = nn.Conv2d(in_channels=self.block3_channel, out_channels=self.block3_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block3_bn5_3 = nn.BatchNorm2d(self.block3_channel* 4)
        # ------------------------------------------------------------------------------
        self.block3_conv6_1 = nn.Conv2d(in_channels=self.block3_channel * 4, out_channels=self.block3_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block3_bn6_1 = nn.BatchNorm2d(self.block3_channel)
        # -----------------------------------------
        self.block3_conv6_2 = nn.Conv2d(in_channels=self.block3_channel, out_channels=self.block3_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block3_bn6_2 = nn.BatchNorm2d(self.block3_channel)
        # -----------------------------------------
        self.block3_conv6_3 = nn.Conv2d(in_channels=self.block3_channel, out_channels=self.block3_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block3_bn6_3 = nn.BatchNorm2d(self.block3_channel* 4)
        # ------------------------------------------------------------------------------

        ###################################block4########################################
        self.block4_downsample = nn.Sequential(
                nn.Conv2d(self.block3_channel*4, self.block4_channel*4, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(self.block4_channel*4))
        self.block4_conv1_1 = nn.Conv2d(in_channels=self.block3_channel*4, out_channels=self.block4_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block4_bn1_1 = nn.BatchNorm2d(self.block4_channel)
        # -----------------------------------------
        self.block4_conv1_2 = nn.Conv2d(in_channels=self.block4_channel, out_channels=self.block4_channel,
                               kernel_size=3, stride=2, bias=False, padding=1)
        self.block4_bn1_2 = nn.BatchNorm2d(self.block4_channel)
        # -----------------------------------------
        self.block4_conv1_3 = nn.Conv2d(in_channels=self.block4_channel, out_channels=self.block4_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block4_bn1_3 = nn.BatchNorm2d(self.block4_channel * 4)
        # ------------------------------------------------------------------------------
        self.block4_conv2_1 = nn.Conv2d(in_channels=self.block4_channel* 4, out_channels=self.block4_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block4_bn2_1 = nn.BatchNorm2d(self.block4_channel)
        # -----------------------------------------
        self.block4_conv2_2 = nn.Conv2d(in_channels=self.block4_channel, out_channels=self.block4_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block4_bn2_2 = nn.BatchNorm2d(self.block4_channel)
        # -----------------------------------------
        self.block4_conv2_3 = nn.Conv2d(in_channels=self.block4_channel, out_channels=self.block4_channel* 4,
                               kernel_size=1, stride=1, bias=False)
        self.block4_bn2_3 = nn.BatchNorm2d(self.block4_channel* 4)
        # ------------------------------------------------------------------------------
        self.block4_conv3_1 = nn.Conv2d(in_channels=self.block4_channel * 4, out_channels=self.block4_channel,
                               kernel_size=1, stride=1, bias=False)
        self.block4_bn3_1 = nn.BatchNorm2d(self.block4_channel)
        # -----------------------------------------
        self.block4_conv3_2 = nn.Conv2d(in_channels=self.block4_channel, out_channels=self.block4_channel,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.block4_bn3_2 = nn.BatchNorm2d(self.block4_channel)
        # -----------------------------------------
        self.block4_conv3_3 = nn.Conv2d(in_channels=self.block4_channel, out_channels=self.block4_channel * 4,
                               kernel_size=1, stride=1, bias=False)
        self.block4_bn3_3 = nn.BatchNorm2d(self.block4_channel* 4)

        if self.include_top:
            # 自适应平均池化，指定输出（H，W），通道数不变
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # 全连接层
            self.fc = nn.Linear(self.block4_channel * 4, num_classes)

        for m in self.modules():
            # isinstance(object, type)：如果指定对象是指定类型，则isinstance()函数返回True
            # 如果是卷积层
            if isinstance(m, nn.Conv2d):
                # kaiming正态分布初始化，使得Conv2d卷积层反向传播的输出的方差都为1
                # fan_in：权重是通过线性层（卷积或全连接）隐性确定
                # fan_out：通过创建随机矩阵显式创建权重
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def checkpoint1(self,x):
        # 无论哪种ResNet，都需要的静态层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.maxpool(x)

        #block1
        identity = out
        identity =self.block1_downsample(out)
        out = self.block1_conv1_1(out)
        out = self.block1_bn1_1(out)
        out = self.relu(out)

        out = self.block1_conv1_2(out)
        out = self.block1_bn1_2(out)
        out = self.relu(out)

        out = self.block1_conv1_3(out)
        out = self.block1_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        identity = out
        out = self.block1_conv2_1(out)
        out = self.block1_bn2_1(out)
        out = self.relu(out)

        out = self.block1_conv2_2(out)
        out = self.block1_bn2_2(out)
        out = self.relu(out)

        out = self.block1_conv2_3(out)
        out = self.block1_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block1_conv3_1(out)
        out = self.block1_bn3_1(out)
        out = self.relu(out)

        out = self.block1_conv3_2(out)
        out = self.block1_bn3_2(out)
        out = self.relu(out)

        out = self.block1_conv3_3(out)
        out = self.block1_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        return out
    
    def checkpoint2(self,out):
        identity = out
        identity = self.block2_downsample(out)
        out = self.block2_conv1_1(out)
        out = self.block2_bn1_1(out)
        out = self.relu(out)

        out = self.block2_conv1_2(out)
        out = self.block2_bn1_2(out)
        out = self.relu(out)

        out = self.block2_conv1_3(out)
        out = self.block2_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block2_conv2_1(out)
        out = self.block2_bn2_1(out)
        out = self.relu(out)

        out = self.block2_conv2_2(out)
        out = self.block2_bn2_2(out)
        out = self.relu(out)

        out = self.block2_conv2_3(out)
        out = self.block2_bn2_3(out)
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)

        out = self.block2_conv3_2(out)
        out = self.block2_bn3_2(out)
        out = self.relu(out)

        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block2_conv4_1(out)
        out = self.block2_bn4_1(out)
        out = self.relu(out)

        out = self.block2_conv4_2(out)
        out = self.block2_bn4_2(out)
        out = self.relu(out)

        out = self.block2_conv3_3(out)
        out = self.block2_bn4_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        
        return out

    def checkpoint3(self,out):
        identity = out
        identity = self.block3_downsample(out)
        out = self.block3_conv1_1(out)
        out = self.block3_bn1_1(out)
        out = self.relu(out)

        out = self.block3_conv1_2(out)
        out = self.block3_bn1_2(out)
        out = self.relu(out)

        out = self.block3_conv1_3(out)
        out = self.block3_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block3_conv2_1(out)
        out = self.block3_bn2_1(out)
        out = self.relu(out)

        out = self.block3_conv2_2(out)
        out = self.block3_bn2_2(out)
        out = self.relu(out)

        out = self.block3_conv2_3(out)
        out = self.block3_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block3_conv3_1(out)
        out = self.block3_bn3_1(out)
        out = self.relu(out)

        out = self.block3_conv3_2(out)
        out = self.block3_bn3_2(out)
        out = self.relu(out)

        out = self.block3_conv3_3(out)
        out = self.block3_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        identity = out
        out = self.block3_conv4_1(out)
        out = self.block3_bn4_1(out)
        out = self.relu(out)

        out = self.block3_conv4_2(out)
        out = self.block3_bn4_2(out)
        out = self.relu(out)

        out = self.block3_conv3_3(out)
        out = self.block3_bn4_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block3_conv5_1(out)
        out = self.block3_bn5_1(out)
        out = self.relu(out)

        out = self.block3_conv5_2(out)
        out = self.block3_bn5_2(out)
        out = self.relu(out)

        out = self.block3_conv5_3(out)
        out = self.block3_bn5_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block3_conv6_1(out)
        out = self.block3_bn6_1(out)
        out = self.relu(out)

        out = self.block3_conv6_2(out)
        out = self.block3_bn6_2(out)
        out = self.relu(out)

        out = self.block3_conv6_3(out)
        out = self.block3_bn6_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        
        return out


    def checkpoint4(self,out):
        identity = out
        identity = self.block4_downsample(out)
        out = self.block4_conv1_1(out)
        out = self.block4_bn1_1(out)
        out = self.relu(out)

        out = self.block4_conv1_2(out)
        out = self.block4_bn1_2(out)
        out = self.relu(out)

        out = self.block4_conv1_3(out)
        out = self.block4_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block4_conv2_1(out)
        out = self.block4_bn2_1(out)
        out = self.relu(out)

        out = self.block4_conv2_2(out)
        out = self.block4_bn2_2(out)
        out = self.relu(out)

        out = self.block4_conv2_3(out)
        out = self.block4_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block4_conv3_1(out)
        out = self.block4_bn3_1(out)
        out = self.relu(out)

        out = self.block4_conv3_2(out)
        out = self.block4_bn3_2(out)
        out = self.relu(out)

        out = self.block4_conv3_3(out)
        out = self.block4_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        
        if self.include_top:
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out= self.fc(out)
        

      
        return out



    def forward(self,x):
        x = x.requires_grad_(True)
        out = checkpoint(self.checkpoint1,x)
        out = checkpoint(self.checkpoint2,out)
        out = checkpoint(self.checkpoint3,out)
        out = checkpoint(self.checkpoint4,out)
        # out = checkpoint(self.checkpoint5,out)
        # out = checkpoint(self.checkpoint6,out)
        # out = checkpoint(self.checkpoint7,out)
        return out
        

        #block2
        
        # 主分支与shortcut分支数据相加
       


        # print(out.size())
        #block3
       


            #block4
        



