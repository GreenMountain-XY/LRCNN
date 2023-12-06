import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
 ### 做了分段的两行，中间合并了一次
# 定义 VGG16 模型
class VGG16(nn.Module):
    def __init__(self, num_classes=1000,num_channels = 64):
        super(VGG16, self).__init__()
        self.num_channels = num_channels
        # 第一段卷积层
        self.conv1_1 = nn.Conv2d(3, self.num_channels, kernel_size=3, padding=1)
        self.ReLU1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(self.num_channels, self.num_channels, kernel_size=3, padding=1)
        self.ReLU1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二段卷积层
        self.conv2_1 = nn.Conv2d(self.num_channels, self.num_channels * 2, kernel_size=3, padding=1)
        self.ReLU2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(self.num_channels * 2, self.num_channels * 2, kernel_size=3, padding=1)
        self.ReLU2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三段卷积层
        self.conv3_1 = nn.Conv2d(self.num_channels * 2,self.num_channels * 4, kernel_size=3, padding=1)
        self.ReLU3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(self.num_channels * 4, self.num_channels * 4, kernel_size=3, padding=1)
        self.ReLU3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(self.num_channels * 4, self.num_channels * 4, kernel_size=3, padding=1)
        self.ReLU3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第四段卷积层
        self.conv4_1 = nn.Conv2d(self.num_channels * 4, self.num_channels * 8, kernel_size=3, padding=1)
        self.ReLU4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(self.num_channels * 8, self.num_channels * 8, kernel_size=3, padding=1)
        self.ReLU4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(self.num_channels * 8, self.num_channels * 8, kernel_size=3, padding=1)
        self.ReLU4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第五段卷积层
        self.conv5_1 = nn.Conv2d(self.num_channels * 8, self.num_channels * 8, kernel_size=3, padding=1)
        self.ReLU5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(self.num_channels * 8, self.num_channels * 8, kernel_size=3, padding=1)
        self.ReLU5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(self.num_channels * 8, self.num_channels * 8, kernel_size=3, padding=1)
        self.ReLU5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_channels * 8 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    def head1(self,x):
        x = self.conv1_1(x) # 122
        x = x[:,:,:-1,:]
        x =self.ReLU1_1(x) #121

        x = self.conv1_2(x) #121
        x = x[:,:,:-1,:]
        x = self.ReLU1_2(x) #120
        x = self.pool1(x) #60

        
        x = self.conv2_1(x) # 60
        x = x[:,:,:-1,:]
        x =self.ReLU2_1(x) #59

        x = self.conv2_2(x) #59
        x = x[:,:,:-1,:]
        x = self.ReLU2_2(x) #58
        x = self.pool2(x) #29
        
        # x = self.conv3_1(x) # 29
        # x = x[:,:,:-1,:]
        # x =self.ReLU3_1(x) #28
     
        # x = self.conv3_2(x) #28
        # x = x[:,:,:-1,:]
        # x = self.ReLU3_2(x) #27
 
        # x = self.conv3_3(x) #27
        # x = x[:,:,:-1,:]
        # x = self.ReLU3_3(x) #26
        # x = self.pool3(x) #13


        return x
    def tail1(self,x):
        x = self.conv1_1(x) # 122
        x = x[:,:,1:,:]
        x =self.ReLU1_1(x) #121

        x = self.conv1_2(x) #121
        x = x[:,:,1:,:]
        x = self.ReLU1_2(x) #120
        x = self.pool1(x) #60

        
        x = self.conv2_1(x) # 60
        x = x[:,:,1:,:]
        x =self.ReLU2_1(x) #59

        x = self.conv2_2(x) #59
        x = x[:,:,1:,:]
        x = self.ReLU2_2(x) #58
        
        x = self.pool2(x) #29


        # x = self.conv3_1(x) # 29
        # x = x[:,:,1:,:]
        # x =self.ReLU3_1(x) #28
     
        # x = self.conv3_2(x) #28
        # x = x[:,:,1:,:]
        # x = self.ReLU3_2(x) #27
 
        # x = self.conv3_3(x) #27
        # x = x[:,:,1:,:]
        # x = self.ReLU3_3(x) #26
        # x = self.pool3(x) #13


        return x    
    def medium1(self,x):
        x = self.conv1_1(x) # 122
        x = x[:,:,1:-1,:]
        x =self.ReLU1_1(x) #121

        x = self.conv1_2(x) #121
        x = x[:,:,1:-1,:]
        x = self.ReLU1_2(x) #120
        x = self.pool1(x) #60

        
        x = self.conv2_1(x) # 60
        x = x[:,:,1:-1,:]
        x =self.ReLU2_1(x) #59

        x = self.conv2_2(x) #59
        x = x[:,:,1:-1,:]
        x = self.ReLU2_2(x) #58
        x = self.pool2(x) #29

        # x = self.conv3_1(x) # 29
        # x = x[:,:,1:-1,:]
        # x =self.ReLU3_1(x) #28
     
        # x = self.conv3_2(x) #28
        # x = x[:,:,1:-1,:]
        # x = self.ReLU3_2(x) #27
 
        # x = self.conv3_3(x) #27
        # x = x[:,:,1:-1,:]
        # x = self.ReLU3_3(x) #26
        # x = self.pool3(x) #13

        return x

    def head2(self,x):
       

        x = self.conv4_1(x) # 13
        x = x[:,:,:-1,:]
        x =self.ReLU4_1(x) #12

        x = self.conv4_2(x) #12
        x = x[:,:,:-1,:]
        x = self.ReLU4_2(x) #11

        x = self.conv4_3(x) #11
        x = x[:,:,:-1,:]
        x = self.ReLU4_3(x) #10
        x = self.pool4(x) #5
        x = self.conv5_1(x) # 5
        x = x[:,:,:-1,:]
        x =self.ReLU5_1(x) #4

        x = self.conv5_2(x) #4
        x = x[:,:,:-1,:]
        x = self.ReLU5_2(x) #3

        x = self.conv5_3(x) #3
        x = x[:,:,:-1,:]
        x = self.ReLU5_3(x) #2
        # x = self.pool5(x) #1
        return x

    def tail2(self,x):
        # x = self.conv3_1(x) # 29
        # x = x[:,:,1:,:]
        # x =self.ReLU3_1(x) #28

        # x = self.conv3_2(x) #28
        # x = x[:,:,1:,:]
        # x = self.ReLU3_2(x) #27
 
        # x = self.conv3_3(x) #27
        # x = x[:,:,1:,:]
        # x = self.ReLU3_3(x) #26
        # x = self.pool3(x) #13


        x = self.conv4_1(x) # 13
        x = x[:,:,1:,:]
        x =self.ReLU4_1(x) #12

        x = self.conv4_2(x) #12
        x = x[:,:,1:,:]
        x = self.ReLU4_2(x) #11

        x = self.conv4_3(x) #11
        x = x[:,:,1:,:]
        x = self.ReLU4_3(x) #10
        x = self.pool4(x) #5



        x = self.conv5_1(x) # 5
        x = x[:,:,1:,:]
        x =self.ReLU5_1(x) #4

        x = self.conv5_2(x) #4
        x = x[:,:,1:,:]
        x = self.ReLU5_2(x) #3

        x = self.conv5_3(x) #3
        x = x[:,:,1:,:]
        x = self.ReLU5_3(x) #2
        # x = self.pool5(x) #1
        return x
    

    def medium2(self,x):
        # x = self.conv3_1(x) # 29
        # x = x[:,:,1:-1,:]
        # x =self.ReLU3_1(x) #28

        # x = self.conv3_2(x) #28
        # x = x[:,:,1:-1,:]
        # x = self.ReLU3_2(x) #27
 
        # x = self.conv3_3(x) #27
        # x = x[:,:,1:-1,:]
        # x = self.ReLU3_3(x) #26
        # x = self.pool3(x) #13


        x = self.conv4_1(x) # 13
        x = x[:,:,1:-1,:]
        x =self.ReLU4_1(x) #12

        x = self.conv4_2(x) #12
        x = x[:,:,1:-1,:]
        x = self.ReLU4_2(x) #11

        x = self.conv4_3(x) #11
        x = x[:,:,1:-1,:]
        x = self.ReLU4_3(x) #10
        x = self.pool4(x) #5



        x = self.conv5_1(x) # 5
        x = x[:,:,1:-1,:]
        x =self.ReLU5_1(x) #4

        x = self.conv5_2(x) #4
        x = x[:,:,1:-1,:]
        x = self.ReLU5_2(x) #3

        x = self.conv5_3(x) #3
        x = x[:,:,1:-1,:]
        x = self.ReLU5_3(x) #2
        # x = self.pool5(x) #1
        return x
    def forward(self, x):
        # x1 = x[:,:,:82,:]
        # x2 = x[:,:,46:130,:]
        # x3 = x[:,:,94:178,:]
        # x4 = x[:,:,142:,:]

        x1 = x[:,:,:62,:]
        x2 = x[:,:,50:118,:]
        x3 = x[:,:,106:174,:]
        x4 = x[:,:,162:,:]
        x1.requires_grad_(True)
        x2.requires_grad_(True)
        x3.requires_grad_(True)
        x4.requires_grad_(True)
        x1 = checkpoint(self.head1,x1)
        x2 = checkpoint(self.medium1,x2)
        x3 = checkpoint(self.medium1,x3)
        x4 = checkpoint(self.tail1,x4)
        # print(x1.size())
        # print(x2.size())
        x = torch.cat((x1,x2,x3,x4),dim = 2)
        # print(x.size())
        # x1 = x[:,:,:23,:]
        # x2 = x[:,:,-23:,:]
        # x1.requires_grad_(True)
        # x2.requires_grad_(True)
        # x1 = checkpoint(self.head2,x1)
        # x2 = checkpoint(self.tail2,x2)
        # x = torch.cat((x1,x2),dim = 2)
        # x = self.conv1_1(x) # 122

        # x =self.ReLU1_1(x) #121

        # x = self.conv1_2(x) #121

        # x = self.ReLU1_2(x) #120
        # x = self.pool1(x) #60

        
        # x = self.conv2_1(x) # 60

        # x =self.ReLU2_1(x) #59

        # x = self.conv2_2(x) #59

        # x = self.ReLU2_2(x) #58
        # x = self.pool2(x) #29



        x = self.conv3_1(x) # 29

        x =self.ReLU3_1(x) #28

        x = self.conv3_2(x) #28

        x = self.ReLU3_2(x) #27
 
        x = self.conv3_3(x) #27

        x = self.ReLU3_3(x) #26
        x = self.pool3(x) #13


        x = self.conv4_1(x) # 13

        x =self.ReLU4_1(x) #12

        x = self.conv4_2(x) #12

        x = self.ReLU4_2(x) #11

        x = self.conv4_3(x) #11

        x = self.ReLU4_3(x) #10
        x = self.pool4(x) #5



        x = self.conv5_1(x) # 5

        x =self.ReLU5_1(x) #4

        x = self.conv5_2(x) #4

        x = self.ReLU5_2(x) #3

        x = self.conv5_3(x) #3

        x = self.ReLU5_3(x) #2
        x = self.pool5(x) #1
 
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
 
 
