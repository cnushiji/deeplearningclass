import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""1.在语言处理、语音识别等方面，文档中每句话的长度是不一样的，且每句话的前后是有关系的，类似这样的数据还有很多，如语音数据、翻译的语句等。像这样与先后顺序有关的数据被称之为序列数据。"""
"""2.卷积神经网络的输入大小是固定的，处理语言、语音等数据就不是卷积神经网络的特长了。"""
"""3.对于序列数据，可以使用循环神经网络（Recurrent Natural Network,RNN）,RNN已经成功应用于自然语言处理、语音识别、图片标注、机器翻译等众多时序问题中。"""
"""- RNN基本运行原理:"""
r"""$ \qquad $ 传统神经网络的结构比较简单：输入层-隐藏层-输出层。如图所示:"""
_, col1, _ = st.columns([1, 2, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片4.png', caption='图4')
r"""$ \qquad $ RNN跟传统神经网络最大的区别在于每次都会将前一次的输出结果，带到下一次的隐藏层中，一起训练。如图所示"""
_, col1, _ = st.columns([1, 2, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片5.gif', caption='图5')
"""- RNN基本结构:"""
_, col1, _ = st.columns([1, 2, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片6.png', caption='图6 循环神经网络的结构')
r"""$ \qquad $ 1.输入x、隐含层、输出层、自循环W,自循环直观理解就是神经元之间还有关联。"""
r"""$ \qquad $ 2.U是输入到隐含层的权重矩阵，W是状态到隐含层的权重矩阵，s为状态，V是隐含层到输出层的权重矩阵。"""
_, col1, _ = st.columns([1, 2, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片7.png', caption='图7 循环神经网络的展开结构')
r"""$ \qquad $ Elman循环神经网络:它的共享参数方式是各个时间节点对应的W、U、V都是不变的，这个机制就像卷积神经网络的过滤机制一样，通过这种方法实现参数共享，同时大大降低参数量。"""
_, col1, _ = st.columns([1, 2, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片8.png', caption='图8 循环神经网络使用单层的全连接结构图')
r"""$ \qquad $ 把隐含层再细化，如图8。这个网络在每一时间t有相同的网络结构。"""
st.latex(r"""a_t=f(Ux_t+Wa_{t-1})""")
r"""$ \qquad $ $𝑜_𝑡$是时刻t的输出。例如，如想预测句子的下一个词，它将会是词汇表中的概率向量，$𝑜_𝑡=𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑉𝑎_𝑡)$"""
r"""$ \qquad $ 循环神经网络最大的特点就是隐层状态，它可以捕获一个序列的一些信息。"""
r"""$ \qquad $ 循环神经网络也可像卷积神经网络一样，除可以横向拓展（增加时间步或序列长度），也可以纵向拓展成多层循环神经网络，如图所示："""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片9.jpg', caption='图9 深层循环神经网络')
r"""- 前向传播与随时间反向传播:"""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片10.jpg', caption='图10 RNN沿时间展开后的结构图')
r"""$ \qquad $ RNN前向传播:"""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片11.jpg', caption='图11 RNN前向传播的计算过程')
r"""$ \qquad $ RNN反向传播:"""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片12.jpg', caption='图12 RNN的BPTT计算示意图')
r"""$ \qquad \qquad $ 循环神经网络的反向传播训练算法称为随时间反向传播（Backpropagation Through Time,BPTT）算法。"""
st.latex(r"""L^{<1>}(\hat{y}^t,y^{<t>})=-y^{<t>}log\hat{y}^t+(1-y^{<t>})log(1-\hat{y}^{<t>})""")
st.latex(r"""L(\hat{y},y)=\sum_{i=1}^{T_y}{L^{<t>}(\hat{y}^{<t>},y^{<t>})}""")
r"""$ \qquad \qquad $ $L^{<t>}$为各输入对应的代价函数，$L(\hat{y},y)$为总代价函数。"""

"""- Eg.RNN是如何工作的？"""
r"""$ \qquad $ 假如需要判断用户的说话意图（问天气、问时间、设置闹钟…），用户说“What time is it?”"""
r"""$ \qquad $ 需要先对这句话进行分词："""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片13.gif', caption='图13')
r"""$ \qquad $ 然后按照顺序输入RNN,先将“What”作为RNN的输入，得到输出01。"""
r"""$ \qquad $ 然后，按顺序，将”time”输入到RNN网络，得到输出02。这个过程可以看到，输入“time”的时候，前面“what”的输出也产生了影响（隐藏层中有一半是黑色的）。"""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片14.gif', caption='图14')
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片15.gif', caption='图15')
r"""$ \qquad $ 以此类推，前面所有的输入都对未来的输出产生了影响，可以看到圆形隐藏层中包含了前面所有的颜色。如下图所示："""
r"""$ \qquad $ 当判断意图的时候，只需要最后一层的输出05，如下图所示："""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片16.gif', caption='图16')
_, col1, _ = st.columns([1, 2, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片17.gif', caption='图17')
r"""$ \qquad $ RNN的缺点也比较明显："""
_, col1, _ = st.columns([1, 2, 1])
with col1:
    st.image('./chapter4_NLP/pages/图片/图片18.jpg', caption='图18')
r"""$ \qquad $ 短期的记忆影响较大（如橙色区域），但是长期的记忆影响就很小（如黑色和绿色区域），这就是RNN存在的短期记忆问题。"""
r"""$ \qquad $ 1.RNN有短期记忆问题，无法处理很长的输入序列。"""
r"""$ \qquad $ 2.训练RNN需要投入极大的成本。"""



