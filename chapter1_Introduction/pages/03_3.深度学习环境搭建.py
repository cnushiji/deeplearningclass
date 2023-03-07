import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

tab1, tab2, tab3 = st.tabs(['1.基本的命令行命令', '2.硬件', '3.环境配置'])

with tab1:
    """- Windows系统："""

    r"""$ \qquad $ 目录切换 cd, dir, mkdir ...
        $ \qquad $ 文件操作 copy, move, del
    """
    """- Linux系统："""
    r"""$ \qquad $ 目录切换  cd，ls,...
        $ \qquad $ 文件操作 mv, rm
        $ \qquad $ 进程查看 ps, top, pgrep...
    """
with tab2:
    """- CPU和内存："""
    r"""$ \qquad $ 主流够用即可 """

    """- 显卡："""
    r"""$ \qquad $ 主流是NVIDIA GPU系列，AMD的基本没有什么支持，不用考虑。安装好驱动，Nvidia在Linux系统尚的驱动安装稍微麻烦。
 """

    """- CUDA："""
    r"""$ \qquad $ CUDA是Nvidia退出的并行计算框架，只支持N卡。深度学习必需。"""

with tab3:
    """- 编程语言："""
    r"""$ \qquad $ python是主流 """

    """- 编程IDE："""
    r"""$ \qquad $ pycharm或者Visual Studio"""

    """- 虚拟环境："""
    r"""$ \qquad $ Anaconda创建虚拟环境"""

    """- Anaconda安装："""
    r"""$ \qquad $ Step1 如图1，选all users"""

    r"""$ \qquad $ Step2 将anaconda添加进环境变量"""

    r"""$ \qquad $ Step3 如图3，最后将两个勾去掉"""

    r"""$ \qquad $ 其余均默认"""

    col1, col2, col3 = st.columns([1,1.1,1.05])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片1.png',caption='图1')
    with col2:
        st.image('./chapter1_Introduction/pages/图片/图片2.png',caption='图2')
    with col3:
        st.image('./chapter1_Introduction/pages/图片/图片3.png',caption='图3')

    """- 下载pytorch："""
    r"""$ \qquad $ Step1 开始菜单打开anaconda Navigator→Environments →base⊳ →open Terminal"""

    r"""$ \qquad $ Step2 在Terminal中输入：
    
    conda env list           # 结果显示只有base
    
    conda create –n pytorch  python=3.8
    
    conda activate pytorch
        
    pip list                 # 可看到有pip
    """

    r"""$ \qquad $ Step3 返回anaconda界面，可看到base下方有pytorch,点击pytorch,可看到右侧有python
在页面上方选 not installed → Search packages中输入pytorch →列表中选中pytorch →右下角点击apply → apply
"""

    r"""$ \qquad $ 完成后，选择installed 可以看到pytorch"""

    """- pycharm安装："""
    r"""$ \qquad $ community版本即可"""

    r"""$ \qquad $ 如图4，全局勾选"""

    r"""$ \qquad $ 其余均默认"""
    _ , col1, _ = st.columns([1,1,1])
    with col1:
        st.image('./pages/图片/图片4.png',caption='图4')

    """- pycharm配置Anaconda环境："""
    r"""$ \qquad $ Step1 双击打开pycharm，点击new project，可更改路径，点击create"""

    r"""$ \qquad $ Step2 file→settings →按图6找到并点击右上角齿轮→Add"""

    r"""$ \qquad $ Step3 图7蓝色框，点击倒三角，选择带有anaconda/pytorch的路径；下方勾选make available to all projects；ok即可
"""

    r"""$ \qquad $ Step4 在main.py中输入import torch，点击run不报错，则环境配置完成"""

    r"""$ \qquad $ 注意：整个过程需要较长时间，大家耐心按步骤安装！"""
    col1, col2 = st.columns([1, 1.03])
    col3, col4 = st.columns([1, 1.23])
    with col1:
        st.image('./pages/图片/图片5.png', caption='图5')
    with col2:
        st.image('./pages/图片/图片6.png', caption='图6')
    with col3:
        st.image('./pages/图片/图片7.png', caption='图7')
    with col4:
        st.image('./pages/图片/图片8.png', caption='图8')

