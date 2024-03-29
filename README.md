# MAPPO-Pettingzoo-simple_spread_v3
将MARL算法MAPPO应用在Pettingzoo中的simple_spread_v3任务中

主要参考：https://github.com/Lizhi-sjtu/MARL-code-pytorch
# Requirments
    gym == 0.25.2
    pettingzoo == 1.24.1
    torch == 1.12.0
    …

# simple_spread_v3
https://pettingzoo.farama.org/environments/mpe/simple_spread/

## Step1.运行train.py
等待训练完成，在本地生成model1/...文件夹
## Step2.运行test.py
测试完成，在本地生成result1/...文件夹

![image] https://github.com/DaydayXtt/MAPPO-Pettingzoo-simple_spread_v3/blob/main/result1/gif/out2.gif

### detials
test.py中的相关路径设置在7、8行