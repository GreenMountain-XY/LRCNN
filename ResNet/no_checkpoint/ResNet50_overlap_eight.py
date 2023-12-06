import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
 

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
    def static_head(self,x):
        x = self.conv1(x) 
        x = x[:,:,:-1,:]
        x = self.maxpool(x)
        x = x[:,:,:-1,:]
        return x
    def static_medium(self,x):
        x = self.conv1(x) 
        x = x[:,:,1:-1,:]
        x = self.maxpool(x)
        x = x[:,:,1:-1,:]
        return x
    def static_tail(self,x):
        x = self.conv1(x) 
        x = x[:,:,1:,:]
        x = self.maxpool(x)
        x = x[:,:,1:,:]
        return x
    def static_forward(self,x):
        x1 = x[:,:,:33,:]
        x2 = x[:,:,23:61,:]
        x3 = x[:,:,51:89,:]
        x4 = x[:,:,79:117,:]
        x5 = x[:,:,107:145,:]
        x6 = x[:,:,135:173,:]
        x7 = x[:,:,163:201,:]
        x8 = x[:,:,191:,:]
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        x3.requires_grad_(True)
        x4.requires_grad_(True)
        x5.requires_grad_(True)
        x6.requires_grad_(True)
        x7.requires_grad_(True)
        x8.requires_grad_(True)
        x1 = checkpoint(self.static_head,x1)
        x2 = checkpoint(self.static_medium,x2)
        x3 = checkpoint(self.static_medium,x3)
        x4 = checkpoint(self.static_medium,x4)
        x5 = checkpoint(self.static_medium,x5)
        x6 = checkpoint(self.static_medium,x6)
        x7 = checkpoint(self.static_medium,x7)
        x8 = checkpoint(self.static_tail,x8)
        x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),dim = 2)
        return x

    def block1_head(self,x):
        # x = self.conv1(x) 
        # x = x[:,:,:-1,:]
        # x = self.maxpool(x)
        # x = x[:,:,:-1,:]
        # # print(x.size())d
        #block1
        identity = x #28

        identity =self.block1_downsample(identity) #28
        out = self.block1_conv1_1(x)
        out = self.block1_bn1_1(out) #28
        out = self.relu(out) 

        out = self.block1_conv1_2(out)
        out = out[:,:,:-1,:]
        identity = identity[:,:,:-1,:]
        out = self.block1_bn1_2(out) 
        out = self.relu(out)
        

        out = self.block1_conv1_3(out)
        out = self.block1_bn1_3(out) #27
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out #27]
        out = self.block1_conv2_1(out)
        out = self.block1_bn2_1(out)
        out = self.relu(out)
        out = self.block1_conv2_2(out)
        out = out[:,:,:-1,:]
        identity = identity[:,:,:-1,:]
        out = self.block1_bn2_2(out)
        out = self.relu(out)

        out = self.block1_conv2_3(out)
        out = self.block1_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out

        out = self.block1_conv3_1(out)#26
        out = self.block1_bn3_1(out)
        out = self.relu(out)

        out = self.block1_conv3_2(out)
        out = out[:,:,:-1,:]
        identity = identity[:,:,:-1,:]
        out = self.block1_bn3_2(out)
        out = self.relu(out)
        
        out = self.block1_conv3_3(out)
        out = self.block1_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        return out
    
    def block1_tail(self,x):
        # x = self.conv1(x) 
        # x = x[:,:,1:,:]
        # x = self.maxpool(x)
        # x = x[:,:,1:,:]
        # # print(x.size())
        #block1
        identity = x #28

        identity =self.block1_downsample(identity) #28
        out = self.block1_conv1_1(x)
        out = self.block1_bn1_1(out) #28
        out = self.relu(out) 

        out = self.block1_conv1_2(out)
        out = out[:,:,1:,:]
        identity = identity[:,:,1:,:]
        out = self.block1_bn1_2(out) 
        out = self.relu(out)
        

        out = self.block1_conv1_3(out)
        out = self.block1_bn1_3(out) #27
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out #27]
        out = self.block1_conv2_1(out)
        out = self.block1_bn2_1(out)
        out = self.relu(out)
        out = self.block1_conv2_2(out)
        out = out[:,:,1:,:]
        identity = identity[:,:,1:,:]
        out = self.block1_bn2_2(out)
        out = self.relu(out)

        out = self.block1_conv2_3(out)
        out = self.block1_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out

        out = self.block1_conv3_1(out)#26
        out = self.block1_bn3_1(out)
        out = self.relu(out)

        out = self.block1_conv3_2(out)
        out = out[:,:,1:,:]
        identity = identity[:,:,1:,:]
        out = self.block1_bn3_2(out)
        out = self.relu(out)
        
        out = self.block1_conv3_3(out)
        out = self.block1_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        return out
    
    def block1_medium(self,x):
        # x = self.conv1(x) 
        # x = x[:,:,1:-1,:]
        # print(x.size())
        # x = self.maxpool(x)
        # x = x[:,:,1:-1,:]
        # print(x.size())
        # print(x.size())
        #block1
        identity = x #28

        identity =self.block1_downsample(identity) #28
        out = self.block1_conv1_1(x)
        out = self.block1_bn1_1(out) #28
        out = self.relu(out) 

        out = self.block1_conv1_2(out)
        out = out[:,:,1:-1,:]
        # print(out.size())
        identity = identity[:,:,1:-1,:]
        out = self.block1_bn1_2(out) 
        out = self.relu(out)
        

        out = self.block1_conv1_3(out)
        out = self.block1_bn1_3(out) #27
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out #27]
        out = self.block1_conv2_1(out)
        out = self.block1_bn2_1(out)
        out = self.relu(out)
        out = self.block1_conv2_2(out)
        out = out[:,:,1:-1,:]
        identity = identity[:,:,1:-1,:]
        out = self.block1_bn2_2(out)
        out = self.relu(out)

        out = self.block1_conv2_3(out)
        out = self.block1_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out

        out = self.block1_conv3_1(out)#26
        out = self.block1_bn3_1(out)
        out = self.relu(out)

        out = self.block1_conv3_2(out)
        out = out[:,:,1:-1,:]
        identity = identity[:,:,1:-1,:]
        out = self.block1_bn3_2(out)
        out = self.relu(out)
        
        out = self.block1_conv3_3(out)
        out = self.block1_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        return out

    def block2_head(self,x):
        #block1
        identity = x #28 
        # identity1 = identity[:,:,-1:,:].clone()
        identity =self.block2_downsample(identity) #28
        out = self.block2_conv1_1(x)
        out = self.block2_bn1_1(out) #28
        out = self.relu(out) 
        # bound1 = out[:,:,-2:,:].clone()
        out = self.block2_conv1_2(out)
        out = out[:,:,:-1,:]#27
        identity = identity[:,:,:-1,:]
        out = self.block2_bn1_2(out) 
        out = self.relu(out)
        

        out = self.block2_conv1_3(out)
        out = self.block2_bn1_3(out) #27
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out #27
        
        out = self.block2_conv2_1(out)
        out = self.block2_bn2_1(out)
        out = self.relu(out)
  
        out = self.block2_conv2_2(out)
        out = out[:,:,:-1,:] #26
        identity = identity[:,:,:-1,:]
        out = self.block2_bn2_2(out)
        out = self.relu(out)
        

        out = self.block2_conv2_3(out)
        out = self.block2_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block2_conv3_1(out)#26
        out = self.block2_bn3_1(out)
        out = self.relu(out)

        out = self.block2_conv3_2(out)
        out = out[:,:,:-1,:]
        identity = identity[:,:,:-1,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
        
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block2_conv3_1(out)#26
        out = self.block2_bn3_1(out)
        out = self.relu(out)
        out = self.block2_conv3_2(out)
        out = out[:,:,:-1,:]
        identity = identity[:,:,:-1,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
        
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        return out
    def block2_medium(self,x):
        identity = x # 28
        identity =self.block2_downsample(identity)
        # print(identity.size())
        out = self.block2_conv1_1(x)
        out = self.block2_bn1_1(out)
        out = self.relu(out)
        out = self.block2_conv1_2(out)
        out = out[:,:,1:-1,:]
        identity = identity[:,:,1:-1,:]
        out = self.block2_bn1_2(out)
        out = self.relu(out)
        # print(out.size())
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
        out = out[:,:,1:-1,:]
        identity = identity[:,:,1:-1,:]
        out = self.block2_bn2_2(out)
        out = self.relu(out)
       
        out = self.block2_conv2_3(out)
        out = self.block2_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)
        out = self.block2_conv3_2(out)
        out = out[:,:,1:-1,:]
        identity = identity[:,:,1:-1,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
       
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)


        identity = out

        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)

        out = self.block2_conv3_2(out)
        out = out[:,:,1:-1,:]
        identity = identity[:,:,1:-1,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
       
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        return out
    def block2_tail(self, x):
        #block1
        identity = x # 28\
        identity =self.block2_downsample(identity)
        out = self.block2_conv1_1(x)
        out = self.block2_bn1_1(out)
        out = self.relu(out)

        out = self.block2_conv1_2(out)
        out = out[:,:,1:,:]
        identity = identity[:,:,1:,:]
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
        out = out[:,:,1:,:]
        identity = identity[:,:,1:,:]
        out = self.block2_bn2_2(out)
        out = self.relu(out)
       
        out = self.block2_conv2_3(out)
        out = self.block2_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out

        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)

        out = self.block2_conv3_2(out)
        out = out[:,:,1:,:]
        identity = identity[:,:,1:,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
       
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
    
    

        identity = out

        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)

        out = self.block2_conv3_2(out)
        out = out[:,:,1:,:]
        identity = identity[:,:,1:,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
       
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        return out
        
    def block2_forward(self,x):
        x1 = x[:,:,:22,:]
        x2 = x[:,:,6:36,:]
        x3 = x[:,:,20:50,:]
        x4 = x[:,:,34:,:]
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        x3.requires_grad_(True)
        x4.requires_grad_(True)
        x1 =checkpoint(self.block2_head,x1)
        x2 =checkpoint(self.block2_medium,x2)
        x3 =checkpoint(self.block2_medium,x3)
        x4 = checkpoint(self.block2_tail,x4)
        x = torch.cat((x1,x2,x3,x4),dim=2)
        return x
    

    
    def forward(self,x):
        # 无论哪种ResNet，都需要的静态层
        x = self.static_forward(x)
  
        # #block1
        # identity = out
        # identity =self.block1_downsample(out)
        # out = self.block1_conv1_1(out)
        # out = self.block1_bn1_1(out)
        # out = self.relu(out)

        # out = self.block1_conv1_2(out)
        # out = self.block1_bn1_2(out)
        # out = self.relu(out)

        # out = self.block1_conv1_3(out)
        # out = self.block1_bn1_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # identity = out
        # out = self.block1_conv2_1(out)
        # out = self.block1_bn2_1(out)
        # out = self.relu(out)

        # out = self.block1_conv2_2(out)
        # out = self.block1_bn2_2(out)
        # out = self.relu(out)

        # out = self.block1_conv2_3(out)
        # out = self.block1_bn2_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # identity = out
        # out = self.block1_conv3_1(out)
        # out = self.block1_bn3_1(out)
        # out = self.relu(out)

        # out = self.block1_conv3_2(out)
        # out = self.block1_bn3_2(out)
        # out = self.relu(out)

        # out = self.block1_conv3_3(out)
        # out = self.block1_bn3_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)
        x1 = x[:,:,:10,:]
        x2 = x[:,:,4:17,:]
        x3 = x[:,:,11:24,:]
        x4 = x[:,:,18:31,:]
        x5 = x[:,:,25:38,:]
        x6 = x[:,:,32:45,:]
        x7 = x[:,:,39:52,:]
        x8 = x[:,:,46:,:]


        
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        x3.requires_grad_(True)
        x4.requires_grad_(True)
        x5.requires_grad_(True)
        x6.requires_grad_(True)
        x7.requires_grad_(True)
        x8.requires_grad_(True)
        # x1.requires_grad_(True)
        # x2.requires_grad_(True)


        x1 = checkpoint(self.block1_head,x1)
        x2 = checkpoint(self.block1_medium,x2)
        x3 = checkpoint(self.block1_medium,x3)
        x4 = checkpoint(self.block1_medium,x4)
        x5 = checkpoint(self.block1_medium,x5)
        x6 = checkpoint(self.block1_medium,x6)
        x7 = checkpoint(self.block1_medium,x7)
        x8 = checkpoint(self.block1_tail,x8)
        # print(x1.size())
        # print(x2.size())
        out = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),dim = 2)
        
        # out = self.block2_forward(out)
        # out = self.block3_forward(out)

        # print(out.size())
        #block2
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
        # 主分支与shortcut分支数据相加
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


        # print(out.size())
        #block3
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

            #block4
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







































