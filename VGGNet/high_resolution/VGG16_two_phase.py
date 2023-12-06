import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
 
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
        self.pool5 = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(self.num_channels * 8 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    

    def head(self, x):
        # 121
        x = self.conv1_1(x) # 122
        x = x[:,:, :-1, :] #121
        x =self.ReLU1_1(x) #121
        bound1_1 = x[: , :, -2:,:].clone() 
        x = self.conv1_2(x) #121
        x = x[:,:, :-1, :] # 120
        x = self.ReLU1_2(x) #120
        # print(1,type(x))
        x = self.pool1(x) #60
        # print(2,type(x))
        bound1_2 = x[: , :, -2:,:].clone()


        # print(3,type(x))
        x = self.conv2_1(x) # 60
        x = x[:,:, :-1, :] #59
        x =self.ReLU2_1(x) #59
        bound2_1 = x[: , :, -2:,:].clone() 
        x = self.conv2_2(x) #59
        x = x[:,:, :-1, :] # 58
        x = self.ReLU2_2(x) #58
        x = self.pool2(x) #29
        bound2_2 = x[: , :, -2:,:].clone()


        x = self.conv3_1(x) # 29
        x = x[:,:, :-1, :] #28
        x =self.ReLU3_1(x) #28
        bound3_1 = x[: , :, -2:,:].clone()
        x = self.conv3_2(x) #28
        x = x[:,:, :-1, :] # 27
        x = self.ReLU3_2(x) #27
        bound3_2 = x[: , :, -2:,:].clone()
        x = self.conv3_3(x) #27
        x = x[:,:, :-1, :] # 26
        x = self.ReLU3_3(x) #26
        x = self.pool3(x) #13
        # bound3_3 = x[: , :, -2:,:].clone()

        # x = self.conv4_1(x) # 13
        # x = x[:,:, :-1, :] #12
        # x =self.ReLU4_1(x) #12
        # bound4_1 = x[: , :, -2:,:].clone()
        # x = self.conv4_2(x) #12
        # x = x[:,:, :-1, :] # 11
        # x = self.ReLU4_2(x) #11
        # bound4_2 = x[: , :, -2:,:].clone()
        # x = self.conv4_3(x) #11
        # x = x[:,:, :-1, :] # 10
        # x = self.ReLU4_3(x) #10
        # x = self.pool4(x) #5
        # bound4_3 = x[: , :, -2:,:].clone()


        # x = self.conv5_1(x) # 5
        # x = x[:,:, :-1, :] #4
        # x =self.ReLU5_1(x) #4
        # bound5_1 = x[: , :, -2:,:].clone()
        # x = self.conv5_2(x) #4
        # x = x[:,:, :-1, :] # 3
        # x = self.ReLU5_2(x) #3
        # bound5_2 = x[: , :, -2:,:].clone()
        # x = self.conv5_3(x) #3
        # x = x[:,:, :-1, :] # 2
        # x = self.ReLU5_3(x) #2
        # x = self.pool5(x) #1
        return bound1_1, bound1_2, bound2_1, bound2_2, bound3_1, bound3_2, x
    
    def tail(self, bound1_1, bound1_2, bound2_1, bound2_2, bound3_1, bound3_2, x):
        x = self.conv1_1(x) # 122
        x = x[:,:, 1:, :] #121
        x =self.ReLU1_1(x) #121
        x = torch.cat((bound1_1,x), dim= 2)
        x = self.conv1_2(x) #121
        x = x[:,:, 1:, :] #121
        x = self.ReLU1_2(x) #120
        x = self.pool1(x) #60
        x = torch.cat((bound1_2,x), dim= 2)
        
        x = self.conv2_1(x) # 60
        x = x[:,:, 1:, :] #121
        x =self.ReLU2_1(x) #59
        x = torch.cat((bound2_1,x), dim= 2)
        x = self.conv2_2(x) #59
        x = x[:,:, 1:, :] #121
        x = self.ReLU2_2(x) #58
        x = self.pool2(x) #29
        x = torch.cat((bound2_2,x), dim= 2)


        x = self.conv3_1(x) # 29
        x = x[:,:, 1:, :] #121
        x =self.ReLU3_1(x) #28
        x = torch.cat((bound3_1,x), dim= 2)
        x = self.conv3_2(x) #28
        x = x[:,:, 1:, :] #121
        x = self.ReLU3_2(x) #27
        x = torch.cat((bound3_2,x), dim= 2)
        x = self.conv3_3(x) #27
        x = x[:,:, 1:, :] #121
        x = self.ReLU3_3(x) #26
        x = self.pool3(x) #13
        # x = torch.cat((bound3_3,x), dim= 2)

        # x = self.conv4_1(x) # 13
        # x = x[:,:, 1:, :] #121
        # x =self.ReLU4_1(x) #12
        # x = torch.cat((bound4_1,x), dim= 2)
        # x = self.conv4_2(x) #12
        # x = x[:,:, 1:, :] #121
        # x = self.ReLU4_2(x) #11
        # x = torch.cat((bound4_2,x), dim= 2)
        # x = self.conv4_3(x) #11
        # x = x[:,:, 1:, :] #121
        # x = self.ReLU4_3(x) #10
        # x = self.pool4(x) #5
        # x = torch.cat((bound4_3,x), dim= 2)


        # x = self.conv5_1(x) # 5
        # x = x[:,:, 1:, :] #121
        # x =self.ReLU5_1(x) #4
        # x = torch.cat((bound5_1,x), dim= 2)
        # x = self.conv5_2(x) #4
        # x = x[:,:, 1:, :] #121
        # x = self.ReLU5_2(x) #3
        # x = torch.cat((bound5_2,x), dim= 2)
        # x = self.conv5_3(x) #3
        # x = x[:,:, 1:, :] #121
        # x = self.ReLU5_3(x) #2
        # x = self.pool5(x) #1
        return x
 
    def medium(self,bound1_1, bound1_2, bound2_1, bound2_2, bound3_1, bound3_2,x):
        x = self.conv1_1(x) # 122
        x = x[:,:, 1:-1, :] #121
        x =self.ReLU1_1(x) #121
        new_bound1_1 = x[:,:, -2:, :].clone()
        x = torch.cat((bound1_1,x), dim= 2)
        x = self.conv1_2(x) #121
        x = x[:,:, 1:-1, :] #121
        x = self.ReLU1_2(x) #120
        x = self.pool1(x) #60
        new_bound1_2 = x[:,:, -2:, :].clone()
        x = torch.cat((bound1_2,x), dim= 2)
        
        x = self.conv2_1(x) # 60
        x = x[:,:, 1:-1, :] #121
        x =self.ReLU2_1(x) #59
        new_bound2_1 = x[:,:, -2:, :].clone()
        x = torch.cat((bound2_1,x), dim= 2)
        x = self.conv2_2(x) #59
        x = x[:,:, 1:-1, :] #121
        x = self.ReLU2_2(x) #58
        x = self.pool2(x) #29
        new_bound2_2 = x[:,:, -2:, :].clone()
        x = torch.cat((bound2_2,x), dim= 2)


        x = self.conv3_1(x) # 29
        x = x[:,:, 1:-1, :] #121
        x =self.ReLU3_1(x) #28
        new_bound3_1 = x[:,:, -2:, :].clone()
        x = torch.cat((bound3_1,x), dim= 2)
        x = self.conv3_2(x) #28
        x = x[:,:, 1:-1, :] #121
        x = self.ReLU3_2(x) #27
        new_bound3_2 = x[:,:, -2:, :].clone()
        x = torch.cat((bound3_2,x), dim= 2)
        x = self.conv3_3(x) #27
        x = x[:,:, 1:-1, :] #121
        x = self.ReLU3_3(x) #26
        x = self.pool3(x) #13
        # new_bound3_3 = x[:,:, -2:, :].clone()
        # x = torch.cat((bound3_3,x), dim= 2)

        # x = self.conv4_1(x) # 13
        # x = x[:,:, 1:-1, :] #121
        # x =self.ReLU4_1(x) #12
        # new_bound4_1 = x[:,:, -2:, :].clone()
        # x = torch.cat((bound4_1,x), dim= 2)
        # x = self.conv4_2(x) #12
        # x = x[:,:, 1:-1, :] #121
        # x = self.ReLU4_2(x) #11
        # new_bound4_2 = x[:,:, -2:, :].clone()
        # x = torch.cat((bound4_2,x), dim= 2)
        # x = self.conv4_3(x) #11
        # x = x[:,:, 1:-1, :] #121
        # x = self.ReLU4_3(x) #10

        # x = self.pool4(x) #5
        
        # new_bound4_3 = x[: , :, -2:,:].clone()
        # x = torch.cat((bound4_3,x), dim= 2)

        # x = self.conv5_1(x) # 5
        # x = x[:,:, 1:-1, :] #4
        # x =self.ReLU5_1(x) #4
        # new_bound5_1 = x[: , :, -2:,:].clone()
        # x = torch.cat((bound5_1,x), dim= 2)
        # x = self.conv5_2(x) #4
        # x = x[:,:, 1:-1, :] # 3
        # x = self.ReLU5_2(x) #3
        # new_bound5_2 = x[: , :, -2:,:].clone()
        # x = torch.cat((bound5_2,x), dim= 2)
        # x = self.conv5_3(x) #3
        # # x = x[:,:, 1:-1, :]# 2
        # x = self.ReLU5_3(x) #2
        # x = self.pool5(x) #1
    
        return new_bound1_1, new_bound1_2, new_bound2_1, new_bound2_2, new_bound3_1, new_bound3_2, x
    
    def head2(self,x):
        x = self.conv4_1(x) # 13
        x = x[:,:, :-1, :] #12
        x =self.ReLU4_1(x) #12
        bound1_1 = x[: , :, -2:,:].clone()
        x = self.conv4_2(x) #12
        x = x[:,:, :-1, :] # 11
        x = self.ReLU4_2(x) #11
        bound1_2 = x[: , :, -2:,:].clone()
        x = self.conv4_3(x) #11
        x = x[:,:, :-1, :] # 10
        x = self.ReLU4_3(x) #10
        x = self.pool4(x) #5
        bound1_3 = x[: , :, -2:,:].clone()


        x = self.conv5_1(x) # 5
        x = x[:,:, :-1, :] #4
        x =self.ReLU5_1(x) #4
        bound2_1 = x[: , :, -2:,:].clone()
        x = self.conv5_2(x) #4
        x = x[:,:, :-1, :] # 3
        x = self.ReLU5_2(x) #3
        bound2_2 = x[: , :, -2:,:].clone()
        x = self.conv5_3(x) #3
        x = x[:,:, :-1, :] # 2
        x = self.ReLU5_3(x) #2
        # x = self.pool5(x) #1
        return bound1_1, bound1_2, bound1_3, bound2_1, bound2_2,  x
    
    def tail2(self,bound1_1, bound1_2, bound1_3, bound2_1, bound2_2,x):
        x = self.conv4_1(x) # 13
        x = x[:,:, 1:, :] #121
        x =self.ReLU4_1(x) #12
        x = torch.cat((bound1_1,x), dim= 2)
        x = self.conv4_2(x) #12
        x = x[:,:, 1:, :] #121
        x = self.ReLU4_2(x) #11
        x = torch.cat((bound1_2,x), dim= 2)
        x = self.conv4_3(x) #11
        x = x[:,:, 1:, :] #121
        x = self.ReLU4_3(x) #10
        x = self.pool4(x) #5
        x = torch.cat((bound1_3,x), dim= 2)


        x = self.conv5_1(x) # 5
        x = x[:,:, 1:, :] #121
        x =self.ReLU5_1(x) #4
        x = torch.cat((bound2_1,x), dim= 2)
        x = self.conv5_2(x) #4
        x = x[:,:, 1:, :] #121
        x = self.ReLU5_2(x) #3
        x = torch.cat((bound2_2,x), dim= 2)
        x = self.conv5_3(x) #3
        x = x[:,:, 1:, :] #121
        x = self.ReLU5_3(x) #2
        # x = self.pool5(x) #1
        return x
    
    def medium2(self,bound1_1, bound1_2, bound1_3, bound2_1,bound2_2,x):
        x = self.conv4_1(x) # 13
        x = x[:,:, 1:-1, :] #121
        x =self.ReLU4_1(x) #12
        new_bound1_1 = x[:,:, -2:, :].clone()
        x = torch.cat((bound1_1,x), dim= 2)
        x = self.conv4_2(x) #12
        x = x[:,:, 1:-1, :] #121
        x = self.ReLU4_2(x) #11
        new_bound1_2 = x[:,:, -2:, :].clone()
        x = torch.cat((bound1_2,x), dim= 2)
        x = self.conv4_3(x) #11
        x = x[:,:, 1:-1, :] #121
        x = self.ReLU4_3(x) #10

        x = self.pool4(x) #5
        
        new_bound1_3 = x[: , :, -2:,:].clone()
        x = torch.cat((bound1_3,x), dim= 2)

        x = self.conv5_1(x) # 5
        x = x[:,:, 1:-1, :] #4
        x =self.ReLU5_1(x) #4
        new_bound2_1 = x[: , :, -2:,:].clone()
        x = torch.cat((bound2_1,x), dim= 2)
        x = self.conv5_2(x) #4
        x = x[:,:, 1:-1, :] # 3
        x = self.ReLU5_2(x) #3
        new_bound2_2 = x[: , :, -2:,:].clone()
        x = torch.cat((bound2_2,x), dim= 2)
        x = self.conv5_3(x) #3
        # x = x[:,:, 1:-1, :]# 2
        # x = self.ReLU5_3(x) #2
        return new_bound1_1,new_bound1_2,new_bound1_3,new_bound2_1,new_bound2_2,x

    def forward(self, x):
        offset = 80
        left ,right =0, 82
        xs=[]
        while right < x.size()[2]:
            xs.append(x[:,:,left:right,:])
            left = right-2
            right = right+offset
        if left < x.size()[2]-2:
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
        bound1_1, bound1_2, bound2_1, bound2_2, bound3_1, bound3_2,x1 =checkpoint(self.head,xs[0])
        outs.append(x1)
        for out in xs[1:-1]:
            bound1_1, bound1_2, bound2_1, bound2_2, bound3_1, bound3_2,out =checkpoint(self.medium, bound1_1, bound1_2, bound2_1, bound2_2, bound3_1, bound3_2,out)
            outs.append(out)
      
        out = checkpoint(self.tail, bound1_1, bound1_2, bound2_1, bound2_2, bound3_1, bound3_2,xs[-1])
        outs.append(out)
        out = torch.cat(tuple(outs), dim = 2)
        print(out.size())



        offset = 50
        left ,right =0, 49
        x2s=[]
        while right < out.size()[2]:
            x2s.append(out[:,:,left:right,:])
            left = right - 2
            right = right+offset
        if left < x.size()[2] -2 :
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
        bound1_1, bound1_2, bound1_3, bound2_1,bound2_2,x1 =checkpoint(self.head2,x2s[0])
        outs.append(x1)
        for out in x2s[1:-1]:
            bound1_1, bound1_2, bound1_3, bound2_1,bound2_2,out =checkpoint(self.medium2,bound1_1, bound1_2, bound1_3, bound2_1,bound2_2,out)
            outs.append(out)

        
        
        # bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x2 =checkpoint(self.block1_medium,bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x2)
        # bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x3 =checkpoint(self.block1_medium,bound_static,identity1,identity2,identity3,bound1, bound2, bound3,x3)
        # print(x1.size())
        # print(x1.size())
        out = checkpoint(self.tail2,bound1_1, bound1_2, bound1_3, bound2_1,bound2_2,x2s[-1])
        outs.append(out)
        out = torch.cat(tuple(outs), dim = 2)
        print(out.size())
       
        # x = self.conv5_1(x) # 5
        # print(x.size())
        # x =self.ReLU5_1(x) #4

        # x = self.conv5_2(x) #4

        # x = self.ReLU5_2(x) #3

        # x = self.conv5_3(x) #3

        # x = self.ReLU5_3(x) #2
        x = self.pool5(out) #1
        # print(x.size())
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
 
 
