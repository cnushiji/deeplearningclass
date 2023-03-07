import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""注意力（Attention）机制由Bengio团队于2014年提出并在近年广泛的应用在深度学习中的各个领域，例如在计算机视觉方向用于捕捉图像上的感受野，或者NLP中用于定位关键token或者特征。"""
"""采用Attention机制的原因时考虑到RNN的计算限制是顺序的，也就是说RNN相关算法智能从左向右依次计算或者从右向左依次计算，这种机制带来了两个问题："""
r"""$ \qquad $ 1）时间片t的计算依赖t-1时刻的计算结果，这样限制了模型的并行能力；"""
r"""$ \qquad $ 2）顺序计算的过程中信息会丢失，尽管LSTM等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象，LSTM依旧无能为力。"""
"""- Transformer的提出解决了上面两个问题，"""
r"""$ \qquad $ 1）首先它使用了Attention机制，将序列中的任意两个位置之间的距离缩小为一个常量；"""
r"""$ \qquad $ 2）其次，它不是类似RNN的顺序结构，因此具有更好的并行性，符合现有的GPU框架。"""
"""- 在机器翻译中，Transformer可概括为如图："""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./pages/图片/图片22.png', caption='图22')
"""- Transformer本质上是一个Encoder-Decoder的结构，那么图22可以表示为图23的结构："""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./pages/图片/图片23.png', caption='图23')
"""- Transformer的Encoder和Decoder均由6个block堆叠而成。"""
col1, col2 = st.columns([1, 1])
with col1:
    st.image('./pages/图片/图片24.png', caption='图24')
with col2:
    st.image('./pages/图片/图片25.png', caption='图25')
st.latex(r"""Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V""")
st.latex(r"""FFN(Z)=max(0,ZW_1+b_1)W_2+b_2""")
"""- 输入编码："""
r"""$ \qquad $ 首先通过Word2Vec等词嵌入方法将输入语料转化成特征向量。"""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./pages/图片/图片26.png', caption='图26')
"""$ \qquad $ 输入编码作为一个tensor输入到encoder中"""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./pages/图片/图片27.png', caption='图27')

"""- self-Attention:"""
r"""$ \qquad $ 在self-attention中，每个单词由3个不同的向量，分别是Query向量（Q）,Key向量(K)和Value向量(V),长度均是64.它们是通过3个不同的权值矩阵$𝑊^𝑄,𝑊^𝐾,𝑊^𝑉$得到，其中三个矩阵的尺寸也是相同的。均是512×64。"""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./pages/图片/图片28.png', caption='图28 Q,K,V的计算示例图')
col1, col2 = st.columns([1, 1])
with col1:
    st.image('./pages/图片/图片29.png', caption='图29 Self-Attention计算示例图')
with col2:
    st.image('./pages/图片/图片30.png', caption='图30 Q,K,V的矩阵表示')
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./pages/图片/图片31.png', caption='图31 Self-Attention的矩阵表示')
st.latex(r"""Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V""")

"""- why self-Attention ?"""
r"""$ \qquad $ 1. computational complexity per layer"""
r"""$ \qquad $ 2. the amout of computation can be parallelized """
r"""$ \qquad $ 3. path length between long-range dependencies in the network"""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./pages/图片/图片32.png', caption='图32')
"""- 残差结构:"""
r"""$ \qquad $ Self-attention需要强调的最后一点是其采用了残差网络中的short-cut结构，目的当然是解决深度学习中的退化问题，得到的最终的结果如图："""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./pages/图片/图片33.png', caption='图33')
"""- Multi-Head Attention:"""
r"""$ \qquad $ Multi-Head Attention相当于h个不同的self-attention的集成（ensemble）如图所示："""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./pages/图片/图片34.png', caption='图34')
"""- Encoder-Decoder Attention:"""
r"""$ \qquad $ 在解码器中，Transformer block比编码器中多了个encoder-decoder attention。在encoder-decoder attention中，Q来自于解码器的上一个输出，K和V则来自于编码器的输出。"""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./pages/图片/图片35.png', caption='图35')
"""- Transformer:"""
r"""$ \qquad $ 一个完整可训练的网络结构是encoder和decoder的堆叠（各N各，N=6）,完整的Transformer结构："""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./pages/图片/图片36.png', caption='图36')
"""- 位置编码（Position Embedding）:"""
r"""$ \qquad $ 位置编码会在词向量中加入了单词的位置信息，这样Transformer就能区分不同位置的单词。"""
r"""$ \qquad $ 通常位置编码是一个长度为𝑑_𝑚𝑜𝑑𝑒𝑙的特征向量，这样便于和词向量进行单位加的操作，如图："""
_, col1, _ = st.columns([1, 4, 1])
with col1:
    st.image('./pages/图片/图片37.png', caption='图37')
"""- 优缺点："""
r"""$ \qquad $ 1.优点："""
r"""$ \qquad \qquad $ 1）设计足够创新，抛弃了在NLP中最根本的RNN或CN并取得了非常不错的效果。"""
r"""$ \qquad \qquad $ 2）不仅限于用在NLP的机器翻译领域。"""
r"""$ \qquad $ 2. 缺点："""
r"""$ \qquad \qquad $ 1）粗暴的抛弃RNN和CNN虽然非常炫技，但是它也使模型丧失了捕捉局部特征的能力，RNN+CNN+Transformer的结合可能会带来更好的效果。"""
r"""$ \qquad \qquad $ 2）Transformer失去的位置信息其实在NLP中非常重要，论文中在特征向量中加入了Position Embedding也只是一个权宜之计，并没有改变Transformer结构上的固有缺陷。"""
