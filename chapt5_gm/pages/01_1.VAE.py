import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

tab1, tab2 = st.tabs(['1.AutoEncoder(AE)', '2.变分自编码器(VAE)'])

with tab1:
    """自编码器（Autoencoder，AE），就是一种利用反向传播算法使得输出值等于输入值的神经网络，它现将输入压缩成潜在空间表征，然后将这种表征重构为输出。所以，从本质上来讲，自编码器是一种数据压缩算法，其压缩和解压缩算法都是通过神经网络来实现的。自编码器有如下三个特点："""
    """$ \qquad $ （1）数据相关性。就是指自编码器只能压缩与自己此前训练数据类似的数据，比如说我们使用mnist训练出来的自编码器用来压缩人脸图片，效果肯定会很差。"""
    """$ \qquad $ （2）数据有损性。自编码器在解压时得到的输出与原始输入相比会有信息损失，所以自编码器是一种数据有损的压缩算法。"""
    """$ \qquad $ （3）自动学习性。自动编码器是从数据样本中自动学习的，这意味着很容易对指定类的输入训练出一种特定的编码器，而不需要完成任何新工作。"""
    """构建一个自编码器需要两部分：编码器（Encoder）和解码器（Decoder）。编码器将输入压缩为潜在空间表征，可以用函数f(x)来表示，解码器将潜在空间表征重构为输出，可以用函数g(x)来表示，编码函数f(x)和解码函数g(x)都是神经网络模型。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片1.jpg', caption='图1 AE网络结构')
    """- AE优缺点:"""
    """$ \qquad $ 1. AutoEncoder是从数据集训练出来的，因此它的压缩能力仅适用于与训练样本相似的数据；"""
    """$ \qquad $ 2. AutoEncoder 还要求 encoder 和 decoder 的能力不能太强。极端情况下，它们有能力完全记忆住训练样本，也就是严重的过拟合。"""
    """- AE的应用:"""
    """$ \qquad $ 自编码器通常有两个方面的应用：一是数据去噪，二是为进行可视化而降维。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片2.jpg', caption='图2')
    """- AE的变体:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片3.png', caption='图3')

with tab2:
    """机器学习有两种范式："""
    """$ \qquad $ （1）判别模型：学习p(y|x)，分类模型；"""
    """$ \qquad $ （2）生成模型：学习p(x|y)，回归模型。变分自编码器是一种生成模型。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片4.jpg', caption='图4')
    """- VAE-理论推导:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片5.png', caption='图5')
    """- 重参数化:"""
    """$ \qquad $ 分布q(z|x,ϕ)依赖于参数ϕ。"""
    """$ \qquad $ 重参数化（reparameterization）是实现通过随机变量实现反向传播的一种重要手段。"""
    st.latex(r"""z\sim{N(\mu_I,\sigma_I^2I)} \longrightarrow z=\mu_I+\sigma_I\odot{\epsilon},\epsilon\sim{N(0,I)}""")
    """- VAE-训练过程:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片6.png', caption='图6')
    """- VAE-学习到的隐变量流形:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片7.png', caption='图7')
    """- VAE-网络结构:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片8.png', caption='图8')
    """- VAE的优缺点:"""
    """$ \qquad $ 1.优点："""
    """$ \qquad \qquad $ 可对数据进行低维表征学习；"""
    """$ \qquad \qquad $ 对似然的精确近似；"""
    """$ \qquad \qquad $ 可插值生成结果，可了解模型学习到了什么；"""
    """$ \qquad \qquad $ 解纠缠表征；"""
    """$ \qquad $ VAE拥有以下特质："""
    """$ \qquad \qquad $ 生成式模型"""
    """$ \qquad \qquad $ 密度模型"""
    """$ \qquad \qquad $ 隐变量模型"""
    """$ \qquad \qquad $ 可对数据降维"""

    """$ \qquad $ 2.缺点："""
    """$ \qquad \qquad $ 生成结果模糊；"""
    """$ \qquad \qquad $ 后验分布被假设为可分解的高斯分布(协方差为对角阵)，以及解码器假设太强；"""
    """$ \qquad \qquad $ 生成更大规模结果(生成大尺寸图像)仍有待研究；"""
    """$ \qquad \qquad $ 运用KL散度对解纠缠进行研究，目前仍处于研究简单示例的阶段；"""
    """$ \qquad \qquad $ 也许还有更好的表示学习方法，或者能得到更好的样本，或者能得到更好的概率密度估计。"""
    """- 变分自编码器的应用:"""
    """$ \qquad $ 1. 应用在图像生成，包括人脸图像、门牌号图像CIFAR 图像、场景物理模型、分割图像以及从静态图像进行预测等。"""
    """$ \qquad $ 2. 应用在NLP领域，如机器翻译。"""
    """$ \qquad $ 参考：《Variational neural machine translation》EMNLP 2016"""
    """$ \qquad $ 《A Hybrid Convolutional Variational Autoencoder for Text Generation》"""

