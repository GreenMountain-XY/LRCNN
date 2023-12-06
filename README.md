**News!!!**   
**[12/03/2023]** Manuscript `LR-CNN: Lightweight Row-Centric Convolutional Neural Network Training for Memory Reduction` is submitted to ICDE 2024 for under view!!!  
**[08/10/2023]** Code cleanup.   
**[07/13/2023]** The existing ieda of Checkpointing is integrated.   
**[05/28/2023]** The existing idea of abandoning cheap-to-be-reomputed data is integrated.  
**[05/11/2023]** Another example RestNet-50 is tested.  
**[04/29/2023]** Two row partitioning policies are implemented.   
**[03/11/2023]** A prototype row-centric implementation on PyTorch using VGG-16 is tested.  
**[11/17/2022]** This project starts.  

# LR-CNN
LR-CNN is a new Convolutional Neural Network (CNN) implementation on PyTorch, with proimnent memory scalability via a novel Lightweight Row-centric computation design.

## 1. Introduction
The multi-layer CNN is one of the most successful deep learning algorithms. It has been widely used in various domains, especially for image understanding. To address complex datasets and tasks, it is becoming deeper and larger, with hundreds of layers and kernel parameters. Thus, training its network yields strong and ever-growing demands on memory resources. Till now, many works have been presented to break the memory wall, but all of them follow the conventional layer-by-layer computation dataflow. `LR-CNN` explores another path to achieve the memory reduction goal. Its novel row-centric dataflow is very orthogonal to existing works and is compatible with them. 

The LR-CNN project started at Ocean University of China in Nov. 2022. The backbone memobers are **Hangyu Yang** (master student) and **Zhigang Wang** (associate professor). Currently it is implemented on top of the widely used deep learning platform PyTorch. Features of LR-CNN are listed as below.   

* ___Weak Dependency:___ We deeply analyze convolution behaviours in CNN and find the weak dependency feature for intra- and inter-layer computations, instead of traditionally assumed the many-to-many strong dependency.  
* ___Row-centric Dataflow:___ Inspired by the weak dependency, we re-organize convolutions into rows through all convolutional layers in both forward- and backward-propagations. Then a lot of feature maps can be removed to reduce the peak memory usage.    
* ___Row Partitioning:___ We give two partitioning policies with different implementations to cope with the weak dependency between consecutive rows. They feature contributions respectively for low- and high-configured devices.
* ___Lightweight Design:___ Our proposals can work only on the originally provided accelerator, like GPU, without additonal hardwares, like CPU RAM and more physical devices. More imortantly, by integrating it into existing works, we can further enhance the benefit of memory reduction, to reduce the economic investments when processing larger CNNs. We have already made attempts for existing Prioritized Recomputation and Checkpointing techniques.        

## 2. Quick Start
LR-CNN is developed on top of PyTorch. Before running it, some softwares must be installed, which is beyond the scope of this document. 

### 2.1 Requirements
* PyTorch 1.13.1 
* Python 3.9.13 or higher version   

### 2.2 Testing LR-CNN  
We should first prepare the input training samples.     

Second, we can submit a training job using the following commands.  
* __For VGG-16__  
`run VGG16/run_2PS.py`  
The argument represents our method:  
- 2PS: `VGGNet/run_2PS.py` 
- 2PS-H: `VGG16/run_2PS_H.py` 
- OverL: `VGG16/run_OverL.py` 
- OverL-H: `VGG16/run_OverL_H.py` 

* __For ResNet-50:__  
`run ResNet/run_2PS.py`  
The argument represents our method:  
our methods:   
- 2PS: `ResNet/run_2PS.py` 
- 2PS-H: `ResNet/run_2PS_H.py` 
- OverL: `ResNet/run_OverL.py` 
- OverL-H: `ResNet/run_OverL_H.py`   


## 3. Contact  
If you encounter any problem with LRCNN, please feel free to contact yanghangyu3961@stu.ouc.edu.cn and wangzhigang@ouc.edu.cn.

