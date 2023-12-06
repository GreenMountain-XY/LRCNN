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


    def block1_head(self,x):
        x = self.conv1(x)
        x = x[:,:,:-1,:]
        bound_static = x[:,:,-2:,:].clone()
        x = self.maxpool(x)
        x = x[:,:,:-1,:]
        # print(x.size())
        #block1
        identity = x #28
        identity1 = identity[:,:,-1:,:].clone()
        identity =self.block1_downsample(identity[:,:,:-1,:]) #28
        out = self.block1_conv1_1(x)
        out = self.block1_bn1_1(out) #28
        out = self.relu(out) 
        bound1 = out[:,:,-2:,:].clone()
        out = self.block1_conv1_2(out)
        out = out[:,:,:-1,:]#27
        out = self.block1_bn1_2(out) 
        out = self.relu(out)
        

        out = self.block1_conv1_3(out)
        out = self.block1_bn1_3(out) #27
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out #27
        identity2 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block1_conv2_1(out)
        out = self.block1_bn2_1(out)
        out = self.relu(out)
        bound2 = out[:,:,-2:,:].clone()
        out = self.block1_conv2_2(out)
        out = out[:,:,:-1,:] #26
        out = self.block1_bn2_2(out)
        out = self.relu(out)
        

        out = self.block1_conv2_3(out)
        out = self.block1_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity3 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block1_conv3_1(out)#26
        out = self.block1_bn3_1(out)
        out = self.relu(out)
        bound3 = out[:,:,-2:,:].clone()
        out = self.block1_conv3_2(out)
        out = out[:,:,:-1,:]
        out = self.block1_bn3_2(out)
        out = self.relu(out)
        
        out = self.block1_conv3_3(out)
        out = self.block1_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)


        identity = out #28 
        identity2_1 = identity[:,:,-1:,:].clone()
        identity =self.block2_downsample(identity[:,:,:-1,:]) #28
        out = self.block2_conv1_1(out)
        out = self.block2_bn1_1(out) #28
        out = self.relu(out) 
        bound2_1 = out[:,:,-2:,:].clone()
        out = self.block2_conv1_2(out)
        out = out[:,:,:-1,:]#27
        out = self.block2_bn1_2(out) 
        out = self.relu(out)
        

        out = self.block2_conv1_3(out)
        out = self.block2_bn1_3(out) #27
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out #27
        identity2_2 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block2_conv2_1(out)
        out = self.block2_bn2_1(out)
        out = self.relu(out)
        bound2_2 = out[:,:,-2:,:].clone()
        out = self.block2_conv2_2(out)
        out = out[:,:,:-1,:] #26
        out = self.block2_bn2_2(out)
        out = self.relu(out)
        

        out = self.block2_conv2_3(out)
        out = self.block2_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity2_3 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block2_conv3_1(out)#26
        out = self.block2_bn3_1(out)
        out = self.relu(out)
        bound2_3 = out[:,:,-2:,:].clone()
        out = self.block2_conv3_2(out)
        out = out[:,:,:-1,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
        
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity2_4 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block2_conv3_1(out)#26
        out = self.block2_bn3_1(out)
        out = self.relu(out)
        bound2_4 = out[:,:,-2:,:].clone()
        out = self.block2_conv3_2(out)
        out = out[:,:,:-1,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
        
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        return bound_static,identity1,identity2,identity3,identity2_1,identity2_2,identity2_3,identity2_4, bound1,bound2,bound3,bound2_1, bound2_2, bound2_3,bound2_4,out
    def block1_tail(self,bound_static,identity1,identity2,identity3,identity2_1,identity2_2,identity2_3,identity2_4, bound1,bound2,bound3,bound2_1, bound2_2, bound2_3,bound2_4,x):
        x = self.conv1(x)
       
        x = x[:,:,1:,:]
        x = torch.cat((bound_static,x),dim=2)
        x = self.maxpool(x)
        x = x[:,:,1:,:]
        # print(x.size())
        #block1
        identity = x # 28
        identity = torch.cat((identity1,identity),dim=2)
        identity =self.block1_downsample(identity)
        out = self.block1_conv1_1(x)
        out = self.block1_bn1_1(out)
        out = self.relu(out)
        out = torch.cat((bound1, out),dim=2)
        out = self.block1_conv1_2(out)
        out = out[:,:,1:,:]
        out = self.block1_bn1_2(out)
        out = self.relu(out)
        
        out = self.block1_conv1_3(out)
        out = self.block1_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity = torch.cat((identity2,identity),dim=2)
        out = self.block1_conv2_1(out)
        out = self.block1_bn2_1(out)
        out = self.relu(out)
        out = torch.cat((bound2,out),dim=2)
        out = self.block1_conv2_2(out)
        out = out[:,:,1:,:]
        out = self.block1_bn2_2(out)
        out = self.relu(out)
       
        out = self.block1_conv2_3(out)
        out = self.block1_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity = torch.cat((identity3,identity),dim=2)
        out = self.block1_conv3_1(out)
        out = self.block1_bn3_1(out)
        out = self.relu(out)
        out = torch.cat((bound3,out),dim = 2)
        out = self.block1_conv3_2(out)
        out = out[:,:,1:,:]
        out = self.block1_bn3_2(out)
        out = self.relu(out)
       
        out = self.block1_conv3_3(out)
        out = self.block1_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out # 28

        identity = torch.cat((identity2_1,identity),dim=2)
        identity =self.block2_downsample(identity)
        # print(identity.size())
        out = self.block2_conv1_1(out)
        out = self.block2_bn1_1(out)
        out = self.relu(out)

        out = torch.cat((bound2_1, out),dim=2)
        out = self.block2_conv1_2(out)
        out = out[:,:,1:,:]
        out = self.block2_bn1_2(out)
        out = self.relu(out)
        # print(out.size())
        out = self.block2_conv1_3(out)
        out = self.block2_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
    
        identity = torch.cat((identity2_2,identity),dim=2)
        out = self.block2_conv2_1(out)
        out = self.block2_bn2_1(out)
        out = self.relu(out)

        out = torch.cat((bound2_2,out),dim=2)
        out = self.block2_conv2_2(out)
        out = out[:,:,1:,:]
        out = self.block2_bn2_2(out)
        out = self.relu(out)
       
        out = self.block2_conv2_3(out)
        out = self.block2_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
  
        identity = torch.cat((identity2_3,identity),dim=2)
        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)
   
        out = torch.cat((bound2_3,out),dim = 2)
        out = self.block2_conv3_2(out)
        out = out[:,:,1:,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
       
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)


        identity = out

        identity = torch.cat((identity2_4,identity),dim=2)
        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)

        out = torch.cat((bound2_4,out),dim = 2)
        out = self.block2_conv3_2(out)
        out = out[:,:,1:,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
       
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        return out
    def block1_medium(self,bound_static,identity1,identity2,identity3,identity2_1,identity2_2,identity2_3,identity2_4, bound1,bound2,bound3,bound2_1, bound2_2, bound2_3,bound2_4,x):
        x = self.conv1(x)
        x = x[:,:,1:-1,:]
        new_bound_static = x[:,:,-2:,:].clone()
        x = torch.cat((bound_static,x),dim=2)
        x = self.maxpool(x)
        x = x[:,:,1:-1,:]
        # print(x.size())
         #block1
        identity = x # 28
        new_identity1 = identity1[:,:,-1:,:].clone()
        identity = torch.cat((identity1,identity[:,:,:-1,:]),dim=2)
        identity =self.block1_downsample(identity)
        out = self.block1_conv1_1(x)
        out = self.block1_bn1_1(out)
        out = self.relu(out)
        new_bound1 = out[:,:,-2:,:].clone()
        out = torch.cat((bound1, out),dim=2)
        out = self.block1_conv1_2(out)
        out = out[:,:,1:-1,:]
        out = self.block1_bn1_2(out)
        out = self.relu(out)
        
        out = self.block1_conv1_3(out)
        out = self.block1_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        new_identity2 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity2,identity[:,:,:-1,:]),dim=2)
        out = self.block1_conv2_1(out)
        out = self.block1_bn2_1(out)
        out = self.relu(out)
        new_bound2 = out[:,:,-2:,:].clone()
        out = torch.cat((bound2,out),dim=2)
        out = self.block1_conv2_2(out)
        out = out[:,:,1:-1,:]
        out = self.block1_bn2_2(out)
        out = self.relu(out)
       
        out = self.block1_conv2_3(out)
        out = self.block1_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        new_identity3 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity3,identity[:,:,:-1,:]),dim=2)
        out = self.block1_conv3_1(out)
        out = self.block1_bn3_1(out)
        out = self.relu(out)
        new_bound3 = out[:,:,-2:,:].clone()
        out = torch.cat((bound3,out),dim = 2)
        out = self.block1_conv3_2(out)
        out = out[:,:,1:-1,:]
        out = self.block1_bn3_2(out)
        out = self.relu(out)
       
        out = self.block1_conv3_3(out)
        out = self.block1_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out # 28
        new_identity2_1 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity2_1,identity[:,:,:-1,:]),dim=2)
        identity =self.block2_downsample(identity)
        # print(identity.size())
        out = self.block2_conv1_1(out)
        out = self.block2_bn1_1(out)
        out = self.relu(out)
        new_bound2_1 = out[:,:,-2:,:].clone()
        out = torch.cat((bound2_1, out),dim=2)
        out = self.block2_conv1_2(out)
        out = out[:,:,:-1,:]################azjd
        out = self.block2_bn1_2(out)
        out = self.relu(out)
        # print(out.size())
        out = self.block2_conv1_3(out)
        out = self.block2_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        new_identity2_2 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity2_2,identity[:,:,:-1,:]),dim=2)
        out = self.block2_conv2_1(out)
        out = self.block2_bn2_1(out)
        out = self.relu(out)
        new_bound2_2 = out[:,:,-2:,:].clone()
        out = torch.cat((bound2_2,out),dim=2)
        out = self.block2_conv2_2(out)
        out = out[:,:,1:-1,:]
        out = self.block2_bn2_2(out)
        out = self.relu(out)
       
        out = self.block2_conv2_3(out)
        out = self.block2_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        new_identity2_3 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity2_3,identity[:,:,:-1,:]),dim=2)
        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)
        new_bound2_3 = out[:,:,-2:,:].clone()
        out = torch.cat((bound2_3,out),dim = 2)
        out = self.block2_conv3_2(out)
        out = out[:,:,1:-1,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
       
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)


        identity = out
        new_identity2_4 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity2_4,identity[:,:,:-1,:]),dim=2)
        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)
        new_bound2_4 = out[:,:,-2:,:].clone()
        out = torch.cat((bound2_4,out),dim = 2)
        out = self.block2_conv3_2(out)
        out = out[:,:,1:-1,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
       
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        return new_bound_static,new_identity1,new_identity2,new_identity3,new_identity2_1,new_identity2_2,new_identity2_3,new_identity2_4,new_bound1,new_bound2,new_bound3,new_bound2_1,new_bound2_2,new_bound2_3,new_bound2_4,out
    # def block1_forward(self,x):
    #     x1 = x[:,:,:20,:]
    #     x2 = x[:,:,20:39,:]
    #     x3 = x[:,:,-17:,:]
    #     x1.requires_grad_(True)
    #     x2.requires_grad_(True)
    #     x3.requires_grad_(True)
    #     identity1,identity2,identity3,bound1, bound2, bound3, x1 =checkpoint(self.block1_head,x1)
    #     identity1,identity2,identity3,bound1,bound2,bound3,x2 = checkpoint(self.block1_medium,identity1,identity2,identity3,bound1,bound2,bound3,x2)
    #     x3 = checkpoint(self.block1_tail,identity1,identity2,identity3,bound1,bound2,bound3,x3)
    #     x = torch.cat((x1,x2,x3),dim=2)
    #     return x
    

        x1 = x[:,:,:19,:]
        x2 = x[:,:,-9:,:]
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        identity1,identity2,identity3,identity4,identify5,identify6,bound1, bound2, bound3, bound4,bound5,bound6,x1 =checkpoint(self.block3_head,x1)
        x2 = checkpoint(self.block3_tail,identity1,identity2,identity3,identity4,identify5,identify6,bound1, bound2, bound3, bound4,bound5,bound6,x2)
        x = torch.cat((x1,x2),dim=2)
        return x
    def head2(self,x):
        
###############################
        identity = x
        # print(identity.size())
        identity3_1 = identity[:,:,-1:,:].clone()
        identity =self.block3_downsample(identity[:,:,:-1,:]) 
        # identity =self.block3_downsample(identity) 
        # print(identity.size())
        # print(identity.size())
        out = self.block3_conv1_1(x)
        out = self.block3_bn1_1(out)
        out = self.relu(out)

        bound3_1 = out[:,:,-2:,:].clone()
        out = self.block3_conv1_2(out)
        out = out[:,:,:-1,:]
        # print(out.size())
        out = self.block3_bn1_2(out)
        out = self.relu(out)
        

        out = self.block3_conv1_3(out)
        out = self.block3_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity3_2 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block3_conv2_1(out)
        out = self.block3_bn2_1(out)
        out = self.relu(out)
        bound3_2 = out[:,:,-2:,:].clone()
        out = self.block3_conv2_2(out)
        out = out[:,:,:-1,:]
        out = self.block3_bn2_2(out)
        out = self.relu(out)

        out = self.block3_conv2_3(out)
        out = self.block3_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity3_3 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block3_conv3_1(out)
        out = self.block3_bn3_1(out)
        out = self.relu(out)
        bound3_3 = out[:,:,-2:,:].clone()
        out = self.block3_conv3_2(out)
        out = out[:,:,:-1,:]
        out = self.block3_bn3_2(out)
        out = self.relu(out)

        out = self.block3_conv3_3(out)
        out = self.block3_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity3_4 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block3_conv4_1(out)
        out = self.block3_bn4_1(out)
        out = self.relu(out)
        bound3_4 = out[:,:,-2:,:].clone()
        out = self.block3_conv4_2(out)
        out = out[:,:,:-1,:]
        out = self.block3_bn4_2(out)
        out = self.relu(out)

        out = self.block3_conv3_3(out)
        out = self.block3_bn4_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity3_5 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block3_conv5_1(out)
        out = self.block3_bn5_1(out)
        out = self.relu(out)
        bound3_5 = out[:,:,-2:,:].clone()
        out = self.block3_conv5_2(out)
        out = out[:,:,:-1,:]
        out = self.block3_bn5_2(out)
        out = self.relu(out)

        out = self.block3_conv5_3(out)
        out = self.block3_bn5_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity3_6 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block3_conv6_1(out)
        out = self.block3_bn6_1(out)
        out = self.relu(out)
        bound3_6 = out[:,:,-2:,:].clone()
        out = self.block3_conv6_2(out)
        out = out[:,:,:-1,:]
        out = self.block3_bn6_2(out)
        out = self.relu(out)

        out = self.block3_conv6_3(out)
        out = self.block3_bn6_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        # print(identity.size())
        identity4_1 = identity[:,:,-1:,:].clone()
        identity =self.block4_downsample(identity[:,:,:-1,:]) 
        # identity =self.block4_downsample(identity) 
        print(identity.size())
        # print(identity.size())
        out = self.block4_conv1_1(out)
        out = self.block4_bn1_1(out)
        out = self.relu(out)

        bound4_1 = out[:,:,-2:,:].clone()
        out = self.block4_conv1_2(out)
        # out = out[:,:,:-1,:]
        # print(out.size())
        out = self.block4_bn1_2(out)
        out = self.relu(out)
        

        out = self.block4_conv1_3(out)
        out = self.block4_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        
        identity = out
        identity4_2 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block4_conv2_1(out)
        out = self.block4_bn2_1(out)
        out = self.relu(out)
        bound4_2 = out[:,:,-2:,:].clone()
        out = self.block4_conv2_2(out)
        out = out[:,:,:-1,:]
        out = self.block4_bn2_2(out)
        out = self.relu(out)

        out = self.block4_conv2_3(out)
        out = self.block4_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity4_3 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block4_conv3_1(out)
        out = self.block4_bn3_1(out)
        out = self.relu(out)
        bound4_3 = out[:,:,-2:,:].clone()
        out = self.block4_conv3_2(out)
        out = out[:,:,:-1,:]
        out = self.block4_bn3_2(out)
        out = self.relu(out)

        out = self.block4_conv3_3(out)
        out = self.block4_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        return identity3_1,identity3_2,identity3_3,identity3_4,identity3_5,identity3_6, identity4_1,identity4_2,identity4_3,bound3_1,bound3_2,bound3_3,bound3_4,bound3_5,bound3_6,bound4_1,bound4_2,bound4_3,out
   
   
    def medium2(self,identity3_1,identity3_2,identity3_3,identity3_4,identity3_5,identity3_6, identity4_1,identity4_2,identity4_3,bound3_1,bound3_2,bound3_3,bound3_4,bound3_5,bound3_6,bound4_1,bound4_2,bound4_3,x):
        


        identity = x# 28
        new_identity3_1 = identity3_1[:,:,-1:,:].clone()
        identity = torch.cat((identity3_1,identity[:,:,:-1,:]),dim=2)
        identity =self.block3_downsample(identity)
        # print(identity.size())
        out = self.block3_conv1_1(x)
        out = self.block3_bn1_1(out)
        out = self.relu(out)
        new_bound3_1 = out[:,:,-2:,:].clone()
        out = torch.cat((bound3_1, out),dim=2)
        out = self.block3_conv1_2(out)
        out = out[:,:,:-1,:]
        out = self.block3_bn1_2(out)
        out = self.relu(out)
        # print(out.size())
        out = self.block3_conv1_3(out)
        out = self.block3_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        new_identity3_2 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity3_2,identity[:,:,:-1,:]),dim=2)
        out = self.block3_conv2_1(out)
        out = self.block3_bn2_1(out)
        out = self.relu(out)
        new_bound3_2 = out[:,:,-2:,:].clone()
        out = torch.cat((bound3_2,out),dim=2)
        out = self.block3_conv2_2(out)
        out = out[:,:,1:-1,:]
        out = self.block3_bn2_2(out)
        out = self.relu(out)
       
        out = self.block3_conv2_3(out)
        out = self.block3_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        new_identity3_3 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity3_3,identity[:,:,:-1,:]),dim=2)
        out = self.block3_conv3_1(out)
        out = self.block3_bn3_1(out)
        out = self.relu(out)
        new_bound3_3 = out[:,:,-2:,:].clone()
        out = torch.cat((bound3_3,out),dim = 2)
        out = self.block3_conv3_2(out)
        out = out[:,:,1:-1,:]
        out = self.block3_bn3_2(out)
        out = self.relu(out)
       
        out = self.block3_conv3_3(out)
        out = self.block3_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)


        identity = out
        new_identity3_4 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity3_4,identity[:,:,:-1,:]),dim=2)
        out = self.block3_conv3_1(out)
        out = self.block3_bn3_1(out)
        out = self.relu(out)
        new_bound3_4 = out[:,:,-2:,:].clone()
        out = torch.cat((bound3_4,out),dim = 2)
        out = self.block3_conv3_2(out)
        out = out[:,:,1:-1,:]
        out = self.block3_bn3_2(out)
        out = self.relu(out)
       
        out = self.block3_conv3_3(out)
        out = self.block3_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        new_identity3_5 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity3_5,identity[:,:,:-1,:]),dim=2)
        out = self.block3_conv3_1(out)
        out = self.block3_bn3_1(out)
        out = self.relu(out)
        new_bound3_5 = out[:,:,-2:,:].clone()
        out = torch.cat((bound3_5,out),dim = 2)
        out = self.block3_conv3_2(out)
        out = out[:,:,1:-1,:]
        out = self.block3_bn3_2(out)
        out = self.relu(out)
       
        out = self.block3_conv3_3(out)
        out = self.block3_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)


        identity = out
        new_identity3_6 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity3_6,identity[:,:,:-1,:]),dim=2)
        out = self.block3_conv3_1(out)
        out = self.block3_bn3_1(out)
        out = self.relu(out)
        new_bound3_6 = out[:,:,-2:,:].clone()
        out = torch.cat((bound3_6,out),dim = 2)
        out = self.block3_conv3_2(out)
        out = out[:,:,1:-1,:]
        out = self.block3_bn3_2(out)
        out = self.relu(out)
       
        out = self.block3_conv3_3(out)
        out = self.block3_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)


        identity = out# 28
        new_identity4_1 = identity4_1[:,:,-1:,:].clone()
        identity = torch.cat((identity4_1,identity[:,:,:-1,:]),dim=2)
        identity =self.block4_downsample(identity)
        # print(identity.size())
        out = self.block4_conv1_1(out)
        out = self.block4_bn1_1(out)
        out = self.relu(out)
        new_bound4_1 = out[:,:,-2:,:].clone()
        out = torch.cat((bound4_1, out),dim=2)
        out = self.block4_conv1_2(out)
        out = out[:,:,:-1,:]
        out = self.block4_bn1_2(out)
        out = self.relu(out)
        # print(out.size())
        out = self.block4_conv1_3(out)
        out = self.block4_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        new_identity4_2 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity4_2,identity[:,:,:-1,:]),dim=2)
        out = self.block4_conv2_1(out)
        out = self.block4_bn2_1(out)
        out = self.relu(out)
        new_bound4_2 = out[:,:,-2:,:].clone()
        out = torch.cat((bound4_2,out),dim=2)
        out = self.block4_conv2_2(out)
        out = out[:,:,1:-1,:]
        out = self.block4_bn2_2(out)
        out = self.relu(out)
       
        out = self.block4_conv2_3(out)
        out = self.block4_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        new_identity4_3 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity4_3,identity[:,:,:-1,:]),dim=2)
        out = self.block4_conv3_1(out)
        out = self.block4_bn3_1(out)
        out = self.relu(out)
        new_bound4_3 = out[:,:,-2:,:].clone()
        out = torch.cat((bound4_3,out),dim = 2)
        out = self.block4_conv3_2(out)
        out = out[:,:,1:-1,:]
        out = self.block4_bn3_2(out)
        out = self.relu(out)
       
        out = self.block4_conv3_3(out)
        out = self.block4_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        return new_identity3_1,new_identity3_2,new_identity3_3,new_identity3_4,new_identity3_5,new_identity3_6,new_identity4_1,new_identity4_2,new_identity4_3,new_bound3_1,new_bound3_2,new_bound3_3,new_bound3_4,new_bound3_5,new_bound3_6,new_bound4_1,new_bound4_2,new_bound4_3,out
    
    def tail2(self,identity3_1,identity3_2,identity3_3,identity3_4,identity3_5,identity3_6, identity4_1,identity4_2,identity4_3,bound3_1,bound3_2,bound3_3,bound3_4,bound3_5,bound3_6,bound4_1,bound4_2,bound4_3,x):
        

        identity = x 

        identity = torch.cat((identity3_1,identity),dim=2)
        identity =self.block3_downsample(identity)
        # print(identity.size())
        out = self.block3_conv1_1(x)
        out = self.block3_bn1_1(out)
        out = self.relu(out)

        out = torch.cat((bound3_1, out),dim=2)
        out = self.block3_conv1_2(out)
        out = out[:,:,1:,:]
        out = self.block3_bn1_2(out)
        out = self.relu(out)
        # print(out.size())
        out = self.block3_conv1_3(out)
        out = self.block3_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out

        identity = torch.cat((identity3_2,identity),dim=2)
        out = self.block3_conv2_1(out)
        out = self.block3_bn2_1(out)
        out = self.relu(out)

        out = torch.cat((bound3_2,out),dim=2)
        out = self.block3_conv2_2(out)
        out = out[:,:,1:,:]
        out = self.block3_bn2_2(out)
        out = self.relu(out)
       
        out = self.block3_conv2_3(out)
        out = self.block3_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out

        identity = torch.cat((identity3_3,identity),dim=2)
        out = self.block3_conv3_1(out)
        out = self.block3_bn3_1(out)
        out = self.relu(out)

        out = torch.cat((bound3_3,out),dim = 2)
        out = self.block3_conv3_2(out)
        out = out[:,:,1:,:]
        out = self.block3_bn3_2(out)
        out = self.relu(out)
       
        out = self.block3_conv3_3(out)
        out = self.block3_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)


        identity = out

        identity = torch.cat((identity3_4,identity),dim=2)
        out = self.block3_conv3_1(out)
        out = self.block3_bn3_1(out)
        out = self.relu(out)
        out = torch.cat((bound3_4,out),dim = 2)
        out = self.block3_conv3_2(out)
        out = out[:,:,1:,:]
        out = self.block3_bn3_2(out)
        out = self.relu(out)
       
        out = self.block3_conv3_3(out)
        out = self.block3_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out

        identity = torch.cat((identity3_5,identity),dim=2)
        out = self.block3_conv3_1(out)
        out = self.block3_bn3_1(out)
        out = self.relu(out)

        out = torch.cat((bound3_5,out),dim = 2)
        out = self.block3_conv3_2(out)
        out = out[:,:,1:,:]
        out = self.block3_bn3_2(out)
        out = self.relu(out)
       
        out = self.block3_conv3_3(out)
        out = self.block3_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)


        identity = out

        identity = torch.cat((identity3_6,identity),dim=2)
        out = self.block3_conv3_1(out)
        out = self.block3_bn3_1(out)
        out = self.relu(out)

        out = torch.cat((bound3_6,out),dim = 2)
        out = self.block3_conv3_2(out)
        out = out[:,:,1:,:]
        out = self.block3_bn3_2(out)
        out = self.relu(out)
       
        out = self.block3_conv3_3(out)
        out = self.block3_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)



        identity = out 

        identity = torch.cat((identity4_1,identity),dim=2)
        identity =self.block4_downsample(identity)
        # print(identity.size())
        out = self.block4_conv1_1(out)
        out = self.block4_bn1_1(out)
        out = self.relu(out)

        out = torch.cat((bound4_1, out),dim=2)
        out = self.block4_conv1_2(out)
        out = out[:,:,1:,:]
        out = self.block4_bn1_2(out)
        out = self.relu(out)
        # print(out.size())
        out = self.block4_conv1_3(out)
        out = self.block4_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out

        identity = torch.cat((identity4_2,identity),dim=2)
        out = self.block4_conv2_1(out)
        out = self.block4_bn2_1(out)
        out = self.relu(out)

        out = torch.cat((bound4_2,out),dim=2)
        out = self.block4_conv2_2(out)
        out = out[:,:,1:,:]
        out = self.block4_bn2_2(out)
        out = self.relu(out)
       
        out = self.block4_conv2_3(out)
        out = self.block4_bn2_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out

        identity = torch.cat((identity4_3,identity),dim=2)
        out = self.block4_conv3_1(out)
        out = self.block4_bn3_1(out)
        out = self.relu(out)

        out = torch.cat((bound4_3,out),dim = 2)
        out = self.block4_conv3_2(out)
        out = out[:,:,1:,:]
        out = self.block4_bn3_2(out)
        out = self.relu(out)
       
        out = self.block4_conv3_3(out)
        out = self.block4_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)


        return out

    
    def forward(self,x):
        # 无论哪种ResNet，都需要的静态层s
        # x1 = x[:,:,:,:]
        # x2 = x[:,:,:,:]
        # x3 = x[:,:,:,:]
        # x4 = x[:,:,:,:]
        # x5 = x[:,:,:,:]
        # x6 = x[:,:,:,:]
        # x7 = x[:,:,:,:]
        # x8 = x[:,:,:,:]
        offset = 47
        left ,right =0, 53
        xs=[]
        while right < x.size()[2]:
            xs.append(x[:,:,left:right,:])
            left = right-4
            right = right+offset
        if left < x.size()[2]-4:
            xs.append(x[:,:,left:,:])
        # x1 = x[:,:,:59,:]
        # x2 = x[:,:,54:115,:]
        # x3 = x[:,:,111:172,:]
        # x4 = x[:,:,168:,:]
        # x1.requires_grad_(True)
        # x2.requires_grad_(True)
        # x3.requires_grad_(True)
        # x4.requires_grad_(True)
        xs[0].requires_grad_(True)
        outs = []
        bound_static,identity1,identity2,identity3,identity2_1,identity2_2,identity2_3,identity2_4, bound1,bound2,bound3,bound2_1, bound2_2, bound2_3,bound2_4,x1 =checkpoint(self.block1_head,xs[0])
        outs.append(x1)
        for out in xs[1:-1]:
            bound_static,identity1,identity2,identity3,identity2_1,identity2_2,identity2_3,identity2_4, bound1,bound2,bound3,bound2_1, bound2_2, bound2_3,bound2_4,out =checkpoint(self.block1_medium,bound_static,identity1,identity2,identity3,identity2_1,identity2_2,identity2_3,identity2_4, bound1,bound2,bound3,bound2_1, bound2_2, bound2_3,bound2_4,out)
            outs.append(out)

        

        # bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x2 =checkpoint(self.block1_medium,bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x2)
        # bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x3 =checkpoint(self.block1_medium,bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x3)
        # print(x1.size())
        # print(x1.size())
        out = checkpoint(self.block1_tail,bound_static,identity1,identity2,identity3,identity2_1,identity2_2,identity2_3,identity2_4, bound1,bound2,bound3,bound2_1, bound2_2, bound2_3,bound2_4,xs[-1])
        outs.append(out)
        out = torch.cat(tuple(outs), dim = 2)
        print(out.size())
        # print(x1.size())
        # print(x2.size())
        # print(x3.size())
        # print(x4.size())
        # out = torch.cat((x1,x2,x3,x4),dim=2)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # out = self.maxpool(x)




        # print(out.size())
        # #block1
        # identity = out
        # identity =self.block1_downsample(identity)
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
        # out = self.block1_forward(out)
        # out = self.block2_forward(out)
        # out = self.block3_forward(out)
        # #block2
        # identity = out
        # identity = self.block2_downsample(identity)
        # out = self.block2_conv1_1(out)
        # out = self.block2_bn1_1(out)
        # out = self.relu(out)

        # out = self.block2_conv1_2(out)
        # out = self.block2_bn1_2(out)
        # out = self.relu(out)

        # out = self.block2_conv1_3(out)
        # out = self.block2_bn1_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # identity = out
        # out = self.block2_conv2_1(out)
        # out = self.block2_bn2_1(out)
        # out = self.relu(out)

        # out = self.block2_conv2_2(out)
        # out = self.block2_bn2_2(out)
        # out = self.relu(out)

        # out = self.block2_conv2_3(out)
        # out = self.block2_bn2_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # identity = out
        # out = self.block2_conv3_1(out)
        # out = self.block2_bn3_1(out)
        # out = self.relu(out)

        # out = self.block2_conv3_2(out)
        # out = self.block2_bn3_2(out)
        # out = self.relu(out)

        # out = self.block2_conv3_3(out)
        # out = self.block2_bn3_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # identity = out
        # out = self.block2_conv4_1(out)
        # out = self.block2_bn4_1(out)
        # out = self.relu(out)

        # out = self.block2_conv4_2(out)
        # out = self.block2_bn4_2(out)
        # out = self.relu(out)

        # out = self.block2_conv3_3(out)
        # out = self.block2_bn4_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # # print(out.size())
        # # print(out.size())
        # # block3
        # identity = out
        # identity = self.block3_downsample(identity)
        # out = self.block3_conv1_1(out)
        # out = self.block3_bn1_1(out)
        # out = self.relu(out)

        # out = self.block3_conv1_2(out)
        # out = self.block3_bn1_2(out)
        # out = self.relu(out)

        # out = self.block3_conv1_3(out)
        # out = self.block3_bn1_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # identity = out
        # out = self.block3_conv2_1(out)
        # out = self.block3_bn2_1(out)
        # out = self.relu(out)

        # out = self.block3_conv2_2(out)
        # out = self.block3_bn2_2(out)
        # out = self.relu(out)

        # out = self.block3_conv2_3(out)
        # out = self.block3_bn2_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # identity = out
        # out = self.block3_conv3_1(out)
        # out = self.block3_bn3_1(out)
        # out = self.relu(out)

        # out = self.block3_conv3_2(out)
        # out = self.block3_bn3_2(out)
        # out = self.relu(out)

        # out = self.block3_conv3_3(out)
        # out = self.block3_bn3_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # identity = out
        # out = self.block3_conv4_1(out)
        # out = self.block3_bn4_1(out)
        # out = self.relu(out)

        # out = self.block3_conv4_2(out)
        # out = self.block3_bn4_2(out)
        # out = self.relu(out)

        # out = self.block3_conv3_3(out)
        # out = self.block3_bn4_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # identity = out
        # out = self.block3_conv5_1(out)
        # out = self.block3_bn5_1(out)
        # out = self.relu(out)

        # out = self.block3_conv5_2(out)
        # out = self.block3_bn5_2(out)
        # out = self.relu(out)

        # out = self.block3_conv5_3(out)
        # out = self.block3_bn5_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # identity = out
        # out = self.block3_conv6_1(out)
        # out = self.block3_bn6_1(out)
        # out = self.relu(out)

        # out = self.block3_conv6_2(out)
        # out = self.block3_bn6_2(out)
        # out = self.relu(out)

        # out = self.block3_conv6_3(out)
        # out = self.block3_bn6_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)
        # # out = self.block4_forward(out)
        # # print(out.sizer())

        offset = 55
        left ,right =0, 55
        x2s=[]
        while right < out.size()[2]:
            x2s.append(out[:,:,left:right,:])
            left = right
            right = right+offset
        if left < x.size()[2]:
            x2s.append(out[:,:,left:,:])
        # x1 = x[:,:,:59,:]
        # x2 = x[:,:,54:115,:]
        # x3 = x[:,:,111:172,:]
        # x4 = x[:,:,168:,:]
        # x1.requires_grad_(True)
        # x2.requires_grad_(True)
        # x3.requires_grad_(True)
        # x4.requires_grad_(True)
        x2s[0].requires_grad_(True)
        outs = []
        identity3_1,identity3_2,identity3_3,identity3_4,identity3_5,identity3_6, identity4_1,identity4_2,identity4_3,bound3_1,bound3_2,bound3_3,bound3_4,bound3_5,bound3_6,bound4_1,bound4_2,bound4_3,x1 =checkpoint(self.head2,x2s[0])
        outs.append(x1)
        for out in x2s[1:-1]:
            identity3_1,identity3_2,identity3_3,identity3_4,identity3_5,identity3_6, identity4_1,identity4_2,identity4_3,bound3_1,bound3_2,bound3_3,bound3_4,bound3_5,bound3_6,bound4_1,bound4_2,bound4_3,out =checkpoint(self.medium2,identity3_1,identity3_2,identity3_3,identity3_4,identity3_5,identity3_6, identity4_1,identity4_2,identity4_3,bound3_1,bound3_2,bound3_3,bound3_4,bound3_5,bound3_6,bound4_1,bound4_2,bound4_3,out)
            outs.append(out)

        
        
        # bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x2 =checkpoint(self.block1_medium,bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x2)
        # bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x3 =checkpoint(self.block1_medium,bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x3)
        # print(x1.size())
        # print(x1.size())
        out = checkpoint(self.tail2,identity3_1,identity3_2,identity3_3,identity3_4,identity3_5,identity3_6, identity4_1,identity4_2,identity4_3,bound3_1,bound3_2,bound3_3,bound3_4,bound3_5,bound3_6,bound4_1,bound4_2,bound4_3,x2s[-1])
        outs.append(out)
        out = torch.cat(tuple(outs), dim = 2)
        print(out.size())
            #block4
        # identity = out
        # identity = self.block4_downsample(identity)
        # out = self.block4_conv1_1(out)
        # out = self.block4_bn1_1(out)
        # out = self.relu(out)

        # out = self.block4_conv1_2(out)
        # out = self.block4_bn1_2(out)
        # out = self.relu(out)

        # out = self.block4_conv1_3(out)
        # out = self.block4_bn1_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # identity = out
        # out = self.block4_conv2_1(out)
        # out = self.block4_bn2_1(out)
        # out = self.relu(out)

        # out = self.block4_conv2_2(out)
        # out = self.block4_bn2_2(out)
        # out = self.relu(out)

        # out = self.block4_conv2_3(out)
        # out = self.block4_bn2_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        # identity = out
        # out = self.block4_conv3_1(out)
        # out = self.block4_bn3_1(out)
        # out = self.relu(out)

        # out = self.block4_conv3_2(out)
        # out = self.block4_bn3_2(out)
        # out = self.relu(out)

        # out = self.block4_conv3_3(out)
        # out = self.block4_bn3_3(out)
        # # 主分支与shortcut分支数据相加
        # out += identity
        # out = self.relu(out)

        
        if self.include_top:
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out= self.fc(out)

        return out

