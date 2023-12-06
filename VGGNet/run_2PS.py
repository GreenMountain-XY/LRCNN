import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.optim as optim
from VGG16_CPUOffload import Net
# from VGG16_two_phase_six import VGG16
from no_checkpoint.VGG16_two_phase_six import VGG16
from load_ImageNet import load_ImageNet


train_loader, val_loader, train_dataset, val_dataset = load_ImageNet('dataset',batch_size=8,size=224)
torch.cuda.empty_cache()
import time
t0 = time.time()  
# net = Net(offload_ratio=1, num_channels = 64) 
net = VGG16(num_classes=10,num_channels=64)
# total = sum([param.nelement() for param in net.parameters()])
# print("Number of parameter: %.2fM" % (total / 1e6))  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
loss_function = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr=0.00001)  # 定义优化器

for epoch in range(1):  # loop over the dataset multiple times 训练5轮

    for step, data in enumerate(train_loader, start=0):   # 遍历训练集样本

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
   
        # zero the parameter gradients 将历史损失梯度清0
        optimizer.zero_grad()
        # forward + backward + optimize

        outputs = net(inputs)   # 将训练图片输入到网络进行正向传播
        # print(torch.cuda.max_memory_allocated())
        loss = loss_function(outputs, labels)   # 计算损失 loss_function(预测值, 真实值)
        loss.backward()  
        # print(torch.cuda.max_memory_allocated())        # 将loss反向传播
        optimizer.step()         # 参数更新
        # with torch.no_grad():
        #     predict_y = torch.max(outputs, dim=1)[1] # 取到（概率）最大的类别的index
        #     train_correct += torch.eq(predict_y, labels).sum().item()
        #     train_num += labels.size(0)
        # # print statistics
        # running_loss += loss.item() * labels.size(0)   # 累加loss
        # print(torch.cuda.max_memory_allocated())
        # 以下进行验证
    # if epoch % 10 == 9:    # print every 500 mini-batches 每隔500步打印一次验证信息
    # with torch.no_grad():  # 以下的计算不去计算每个节点的损失梯度
    #     test_correct = 0.0
    #     test_num = 0.0
    #     for val_step, val_data in enumerate(val_loader,start=0):
    #         val_image, val_label = val_data
    #         val_image = val_image.to(device)
    #         val_label = val_label.to(device)
    #         outputs = net(val_image)  # [batch, 10] 验证集正向传播
    #         predict_y = torch.max(outputs, dim=1)[1] # 取到（概率）最大的类别的index
    #         test_correct += torch.eq(predict_y, val_label).sum().item()
            # test_num += val_label.size(0)
    # print(f"epoch: {epoch}, training loss: {running_loss/train_num :.3f}, train accuracy: {train_correct/train_num :.3f}, test accuracy:  {test_correct/test_num :.3f}")
    # train_losses.append(running_loss/train_num)
    # train_accuracys.append(train_correct/train_num)
    # test_accuracys.append(test_correct/test_num)
t1 = time.time()
print(t1-t0)
print(f"The maximum GPU memory usage during the entire training process:{torch.cuda.max_memory_allocated()/(1024 * 1024 * 1024):.3f}GB")
