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
        return bound_static,identity1,identity2,identity3,bound1,bound2,bound3,out
    def block1_tail(self,bound_static,identity1,identity2,identity3,bound1, bound2, bound3, x):
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
        return out
    def block1_medium(self,bound_static, identity1, identity2, identity3, bound1, bound2, bound3, x):
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
        return new_bound_static,new_identity1,new_identity2,new_identity3,new_bound1,new_bound2,new_bound3,out
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
    
    def block2_head(self,x):
        #block1
        identity = x #28 
        identity1 = identity[:,:,-1:,:].clone()
        identity =self.block2_downsample(identity[:,:,:-1,:]) #28
        out = self.block2_conv1_1(x)
        out = self.block2_bn1_1(out) #28
        out = self.relu(out) 
        bound1 = out[:,:,-2:,:].clone()
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
        identity2 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block2_conv2_1(out)
        out = self.block2_bn2_1(out)
        out = self.relu(out)
        bound2 = out[:,:,-2:,:].clone()
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
        identity3 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block2_conv3_1(out)#26
        out = self.block2_bn3_1(out)
        out = self.relu(out)
        bound3 = out[:,:,-2:,:].clone()
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
        identity4 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block2_conv3_1(out)#26
        out = self.block2_bn3_1(out)
        out = self.relu(out)
        bound4 = out[:,:,-2:,:].clone()
        out = self.block2_conv3_2(out)
        out = out[:,:,:-1,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
        
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        return identity1,identity2,identity3,identity4,bound1,bound2,bound3,bound4,out
    def block2_medium(self,identity1,identity2,identity3,identity4,bound1,bound2,bound3,bound4,x):
        identity = x # 28
        new_identity1 = identity1[:,:,-1:,:].clone()
        identity = torch.cat((identity1,identity[:,:,:-1,:]),dim=2)
        identity =self.block2_downsample(identity)
        # print(identity.size())
        out = self.block2_conv1_1(x)
        out = self.block2_bn1_1(out)
        out = self.relu(out)
        new_bound1 = out[:,:,-2:,:].clone()
        out = torch.cat((bound1, out),dim=2)
        out = self.block2_conv1_2(out)
        out = out[:,:,:-1,:]
        out = self.block2_bn1_2(out)
        out = self.relu(out)
        # print(out.size())
        out = self.block2_conv1_3(out)
        out = self.block2_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        new_identity2 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity2,identity[:,:,:-1,:]),dim=2)
        out = self.block2_conv2_1(out)
        out = self.block2_bn2_1(out)
        out = self.relu(out)
        new_bound2 = out[:,:,-2:,:].clone()
        out = torch.cat((bound2,out),dim=2)
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
        new_identity3 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity3,identity[:,:,:-1,:]),dim=2)
        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)
        new_bound3 = out[:,:,-2:,:].clone()
        out = torch.cat((bound3,out),dim = 2)
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
        new_identity4 = identity[:,:,-1:,:].clone()
        identity = torch.cat((identity4,identity[:,:,:-1,:]),dim=2)
        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)
        new_bound4 = out[:,:,-2:,:].clone()
        out = torch.cat((bound4,out),dim = 2)
        out = self.block2_conv3_2(out)
        out = out[:,:,1:-1,:]
        out = self.block2_bn3_2(out)
        out = self.relu(out)
       
        out = self.block2_conv3_3(out)
        out = self.block2_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        return new_identity1,new_identity2,new_identity3,new_identity4,new_bound1,new_bound2,new_bound3,new_bound4,out
    def block2_tail(self,identity1,identity2,identity3,identity4,bound1,bound2,bound3,bound4, x):
        #block1
        identity = x # 28
        identity = torch.cat((identity1,identity),dim=2)
        identity =self.block2_downsample(identity)
        out = self.block2_conv1_1(x)
        out = self.block2_bn1_1(out)
        out = self.relu(out)
        out = torch.cat((bound1, out),dim=2)
        out = self.block2_conv1_2(out)
        out = out[:,:,1:,:]
        out = self.block2_bn1_2(out)
        out = self.relu(out)
        
        out = self.block2_conv1_3(out)
        out = self.block2_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity = torch.cat((identity2,identity),dim=2)
        out = self.block2_conv2_1(out)
        out = self.block2_bn2_1(out)
        out = self.relu(out)
        out = torch.cat((bound2,out),dim=2)
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
        identity = torch.cat((identity3,identity),dim=2)
        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)
        out = torch.cat((bound3,out),dim = 2)
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
        identity = torch.cat((identity4,identity),dim=2)
        out = self.block2_conv3_1(out)
        out = self.block2_bn3_1(out)
        out = self.relu(out)
        out = torch.cat((bound4,out),dim = 2)
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
        
    def block2_forward(self,x):
        x1 = x[:,:,:15,:]
        x2 = x[:,:,15:29,:]
        x3 = x[:,:,29:43,:]
        x4 = x[:,:,-13:,:]
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        x3.requires_grad_(True)
        x4.requires_grad_(True)
        identity1,identity2,identity3,identity4,bound1, bound2, bound3, bound4,x1 =checkpoint(self.block2_head,x1)
        identity1,identity2,identity3,identity4,bound1, bound2, bound3, bound4,x2 =checkpoint(self.block2_medium,identity1,identity2,identity3,identity4,bound1, bound2, bound3, bound4,x2)
        identity1,identity2,identity3,identity4,bound1, bound2, bound3, bound4,x3 =checkpoint(self.block2_medium,identity1,identity2,identity3,identity4,bound1, bound2, bound3, bound4,x3)
        x4 = checkpoint(self.block2_tail,identity1,identity2,identity3,identity4,bound1, bound2, bound3, bound4,x4)
        x = torch.cat((x1,x2,x3,x4),dim=2)
        return x
    

    def block3_head(self,x):
        identity = x
        identity1 = identity[:,:,-1:,:].clone()
        identity =self.block3_downsample(identity[:,:,:-1,:]) 
        # print(identity.size())
        out = self.block3_conv1_1(x)
        out = self.block3_bn1_1(out)
        out = self.relu(out)

        bound1 = out[:,:,-2:,:].clone()
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
        identity2 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block3_conv2_1(out)
        out = self.block3_bn2_1(out)
        out = self.relu(out)
        bound2 = out[:,:,-2:,:].clone()
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
        identity3 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block3_conv3_1(out)
        out = self.block3_bn3_1(out)
        out = self.relu(out)
        bound3 = out[:,:,-2:,:].clone()
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
        identity4 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block3_conv4_1(out)
        out = self.block3_bn4_1(out)
        out = self.relu(out)
        bound4 = out[:,:,-2:,:].clone()
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
        identity5 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block3_conv5_1(out)
        out = self.block3_bn5_1(out)
        out = self.relu(out)
        bound5 = out[:,:,-2:,:].clone()
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
        identity6 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block3_conv6_1(out)
        out = self.block3_bn6_1(out)
        out = self.relu(out)
        bound6 = out[:,:,-2:,:].clone()
        out = self.block3_conv6_2(out)
        out = out[:,:,:-1,:]
        out = self.block3_bn6_2(out)
        out = self.relu(out)

        out = self.block3_conv6_3(out)
        out = self.block3_bn6_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        return identity1,identity2,identity3,identity4,identity5,identity6,bound1,bound2,bound3,bound4,bound5,bound6,out
    
    def block3_tail(self,identity1,identity2,identity3,identity4,identity5,identity6,bound1,bound2,bound3,bound4,bound5,bound6,x):
        identity = x
        identity = torch.cat((identity1,identity),dim=2)
        identity =self.block3_downsample(identity[:,:,:-1,:]) 
        out = self.block3_conv1_1(x)
        out = self.block3_bn1_1(out)
        out = self.relu(out)

        out = torch.cat((bound1, out),dim=2)
        out = self.block3_conv1_2(out)
        out = out[:,:,1:,:]
        out = self.block3_bn1_2(out)
        out = self.relu(out)
        

        out = self.block3_conv1_3(out)
        out = self.block3_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity = torch.cat((identity2,identity),dim=2)
        out = self.block3_conv2_1(out)
        out = self.block3_bn2_1(out)
        out = self.relu(out)
        out = torch.cat((bound2, out),dim=2)
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
        identity = torch.cat((identity3,identity),dim=2)
        out = self.block3_conv3_1(out)
        out = self.block3_bn3_1(out)
        out = self.relu(out)
        out = torch.cat((bound3, out),dim=2)
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
        identity = torch.cat((identity4,identity),dim=2)
        out = self.block3_conv4_1(out)
        out = self.block3_bn4_1(out)
        out = self.relu(out)
        out = torch.cat((bound4, out),dim=2)
        out = self.block3_conv4_2(out)
        out = out[:,:,1:,:]
        out = self.block3_bn4_2(out)
        out = self.relu(out)

        out = self.block3_conv3_3(out)
        out = self.block3_bn4_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity = torch.cat((identity5,identity),dim=2)
        out = self.block3_conv5_1(out)
        out = self.block3_bn5_1(out)
        out = self.relu(out)
        out = torch.cat((bound5, out),dim=2)
        out = self.block3_conv5_2(out)
        out = out[:,:,1:,:]
        out = self.block3_bn5_2(out)
        out = self.relu(out)

        out = self.block3_conv5_3(out)
        out = self.block3_bn5_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity = torch.cat((identity6,identity),dim=2)
        out = self.block3_conv6_1(out)
        out = self.block3_bn6_1(out)
        out = self.relu(out)
        out = torch.cat((bound6, out),dim=2)
        out = self.block3_conv6_2(out)
        out = out[:,:,1:,:]
        out = self.block3_bn6_2(out)
        out = self.relu(out)

        out = self.block3_conv6_3(out)
        out = self.block3_bn6_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)
        return out
    def block3_forward(self,x):
        x1 = x[:,:,:19,:]
        x2 = x[:,:,-9:,:]
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        identity1,identity2,identity3,identity4,identify5,identify6,bound1, bound2, bound3, bound4,bound5,bound6,x1 =checkpoint(self.block3_head,x1)
        x2 = checkpoint(self.block3_tail,identity1,identity2,identity3,identity4,identify5,identify6,bound1, bound2, bound3, bound4,bound5,bound6,x2)
        x = torch.cat((x1,x2),dim=2)
        return x
    

    def block4_head(self,x):
        identity = x
        identity1 = identity[:,:,-1:,:].clone()
        identity =self.block4_downsample(identity[:,:,:-1,:]) 
        out = self.block4_conv1_1(x)
        out = self.block4_bn1_1(out)
        out = self.relu(out)

        bound1 = out[:,:,-2:,:].clone()
        out = self.block4_conv1_2(out)
        out = out[:,:,:-1,:]
        out = self.block4_bn1_2(out)
        out = self.relu(out)
        

        out = self.block4_conv1_3(out)
        out = self.block4_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity2 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block4_conv2_1(out)
        out = self.block4_bn2_1(out)
        out = self.relu(out)
        bound2 = out[:,:,-2:,:].clone()
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
        identity3 = identity[:,:,-1:,:].clone()
        identity = identity[:,:,:-1,:]
        out = self.block4_conv3_1(out)
        out = self.block4_bn3_1(out)
        out = self.relu(out)
        bound3 = out[:,:,-2:,:].clone()
        out = self.block4_conv3_2(out)
        out = out[:,:,:-1,:]
        out = self.block4_bn3_2(out)
        out = self.relu(out)

        out = self.block4_conv3_3(out)
        out = self.block4_bn3_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        
        return identity1,identity2,identity3,bound1,bound2,bound3,out
    def block4_tail(self,identity1,identity2,identity3,bound1,bound2,bound3,x):
        identity = x
        identity = torch.cat((identity1,identity),dim=2)
        identity =self.block4_downsample(identity[:,:,:-1,:]) 
        out = self.block4_conv1_1(x)
        out = self.block4_bn1_1(out)
        out = self.relu(out)

        out = torch.cat((bound1, out),dim=2)
        out = self.block4_conv1_2(out)
        out = out[:,:,1:,:]
        out = self.block4_bn1_2(out)
        out = self.relu(out)
        

        out = self.block4_conv1_3(out)
        out = self.block4_bn1_3(out)
        # 主分支与shortcut分支数据相加
        out += identity
        out = self.relu(out)

        identity = out
        identity = torch.cat((identity2,identity),dim=2)
        out = self.block4_conv2_1(out)
        out = self.block4_bn2_1(out)
        out = self.relu(out)
        out = torch.cat((bound2, out),dim=2)
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
        identity = torch.cat((identity3,identity),dim=2)
        out = self.block4_conv3_1(out)
        out = self.block4_bn3_1(out)
        out = self.relu(out)
        out = torch.cat((bound3, out),dim=2)
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
    def block4_forward(self,x):
        x1 = x[:,:,:11,:]
        x2 = x[:,:,-3:,:]
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        identity1,identity2,identity3,bound1, bound2, bound3,x1 =checkpoint(self.block4_head,x1)
        x2 = checkpoint(self.block4_tail,identity1,identity2,identity3,bound1, bound2, bound3,x2)
        x = torch.cat((x1,x2),dim=2)
        return x
    
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

        x1 = x[:,:,:59,:]
        x2 = x[:,:,54:115,:]
        x3 = x[:,:,111:172,:]
        x4 = x[:,:,168:,:]
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        x3.requires_grad_(True)
        x4.requires_grad_(True)
        bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x1 =checkpoint(self.block1_head,x1)
        bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x2 =checkpoint(self.block1_medium,bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x2)
        bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x3 =checkpoint(self.block1_medium,bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x3)
        # print(x1.size())
        # print(x1.size())
        x4 = checkpoint(self.block1_tail,bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x4)
        # print(x1.size())
        # print(x2.size())
        # print(x3.size())
        # print(x4.size())
        out = torch.cat((x1,x2,x3,x4),dim=2)
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # out = self.maxpool(x)




        print(out.size())
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
        out = self.block2_forward(out)
        print(out.size())
        out = self.block3_forward(out)
        print(out.size())
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

        # print(out.size())
        # # print(out.size())
        # block3
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
        out = self.block4_forward(out)
        # print(out.sizer())
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

