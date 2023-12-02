# 1 概念理解

## 1.1 神经网络训练流程概述

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;当我们使用前馈神经网络（feedfowrward neural network）接收输入 x 并产生输出 y 时，信息通过网络向前流动。输入 x 提供初始信息，然后传播到每一层的隐藏单元，最终产生输出 y。这称之为前向传播（forward propagation）。
在训练过程中，前向传播可以持续向前直到它产生一个**标量** 的 损失函数 $J(\theta)$ 。
反向传播（back propagation）算法经常简称为backprop，允许来自代价函数的信息通过网络向后流动，以便计算梯度。<br>

## 1.2 反向传播的定义
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;反向传播（英语：Backpropagation，意为**误差**反向传播，缩写为BP）是对多层人工神经网络进行梯度下降的算法，也就是用**链式法则**以网络每层的**权重**为变量计算**损失函数**的梯度，以**更新权重**来最小化损失函数。<br>

## 1.3 梯度下降算法简述
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;多元函数 f 的梯度定义为：
$$ gradf=(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y})=\nabla f $$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;梯度有一个非常重要的性质：
- 函数f沿梯度方向增加（上升）最快, 函数f沿负梯度方向减小（下降）最快。

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;因此，我们可以利用梯度的上述性质推导出梯度下降法用于求解一个函数的极小值, ：<br>
![梯度下降法](images/back-propagation-figure1.jpg)

**梯度下降法最小化损失函数展示：**

![梯度下降法](images/back-propagation-gif1.gif)


**梯度下降法代码案例**
```python
#coding:utf8
    
def fun(x,y):
    return x*x + y*y + 2*x +2

def dfun_x(x,y): 
    return 2*x + 2 

def dfun_y(x,y):
    return 2*y

if __name__ == '__main__':    
    x = 1
    y = 4
    lr = 0.01
    iters = 4000

    for iter in range(iters):
        x = x - lr* dfun_x(x, y)
        y = y - lr* dfun_y(x, y)
        print('loss = ', fun(x, y))
        print('x=',x)
        print('y=',y)
```
