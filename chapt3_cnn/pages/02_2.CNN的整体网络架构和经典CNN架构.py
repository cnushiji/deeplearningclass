import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tab1, tab2, tab3 = st.tabs(['1.CNN网络的整体结构', '2.经典的CNN', '3.注意力机制'])

with tab1:
    """- 一个典型的卷积网络是由卷积层、汇聚层、全连接层交叉堆叠而成。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片14.png', caption='图14 常用的卷积网络整体结构')
    """- 目前，卷积网络的整体结构趋向于使用更小的卷积核，以及更深的结构。是否选用全连接层视具体任务而定，比如对图片回归任务来说，通常选用全卷积神经网络。"""

with tab2:
    """- Lenet-5 """
    r""" $ \qquad $ 简介：1998年，由Lecun提出。它是一个非常成功的神经网络模型。基于LeNet-5的手写数字识别系统在 20世纪90年代被美国很多银行使用，用来识别支票上面的手写数字。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片15.png', caption='图15 Lenet-5的网络结构')
    """- AlexNet"""
    r""" $ \qquad $ 简介：AlexNet[Krizhevsky et al., 2012]是第一个现代深度卷积网络模型，其首次使用了很多现代深度卷积网络的技术方法，比如使用 GPU 进行并行训练，采用了 ReLU 作为非线性激活函数，使用 Dropout 防止过拟合，使用数据增强来提高模型准确率等。AlexNet赢得了2012年ImageNet图像分类竞赛的冠军。这些技术极大地推动了端到端的深度学习模型的发展。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片16.jpg', caption='图16 AlexNet的网络结构')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片17.png', caption='图17 AlexNet的网络结构')

    """- VGGNet """
    r""" $ \qquad $ 简介："""
    r""" $ \qquad \qquad $ VGGNet由牛津大学计算机视觉组合和Google DeepMind公司研究员一起研发的深度卷积神经网络。"""
    r""" $ \qquad \qquad $ 它探索了卷积神经网络的深度和其性能之间的关系，通过反复的堆叠$3\times{3}$的小型卷积核和$2\times{2}$的最大池化层，成功的构建了16~19层深的卷积神经网络。"""
    r""" $ \qquad \qquad $ VGGNet获得了ILSVRC 2014年比赛的亚军和定位项目的冠军，在top5上的错误率为7.5%。目前为止，VGGNet依然被用来提取图像的特征。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片18.png', caption='图18 VGGNet的网络结构')
    """- InceptionNet-V1 """
    r""" $ \qquad $ 简介："""
    r""" $ \qquad \qquad $ Google Inception Net(Inception V1)首次出现是在ILSVRC 2014的比赛中，Google Inception Net以top5的错误率为6.67%获得了第一名，而VGGNet的top5错误率为7.3%。Inception V1的特点就是控制了计算量和参数量，Inception V1只有500万 的参数量，而AlexNet有6000万，Inception V1一共有22层，比VGGNet更深。"""
    r""" $ \qquad \qquad $ Inception V1降低参数量主要有两个目的，第一，参数量越大，需要提供模型学习的数据量越大。第二，参数越多，需要消耗的硬件资源更多。"""
    r""" $ \qquad \qquad $ Inception V1参数少但效果好的原因除了模型层数更深、表达能力更强之外，还得益于："""
    r""" $ \qquad \qquad $ 第一，去除了最后的全连接层，使用全局平均池化层来代替全连接层，通过平均池化将图片尺寸变为1×1，在卷积神经网络中，全连接层占据模型的参数量是最多的，使用全连接层还会引起过拟合，去除全连接层可以降低模型过拟合的同时还能加快模型的训练。用全局平均池化层来取代全连接层的思想来自于Network In Network论文。"""
    r""" $ \qquad \qquad $ 第二，在Inception V1中精心设计了Inception Module来提高参数的利用率，Inception Module其实就是大网络中的一个小网络，通过反复堆叠Inception Module来形成一个大网络。"""
    r""" $ \qquad $ Inception模块:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片19.png', caption='图19 Inception模块')
    r""" $ \qquad $ InceptionNet V1的网络结构："""
    r""" $ \qquad \qquad $ Inception 网络有多个版本，其中最早的Inception v1版本就是非常著名的GoogLeNet [Szegedy et al., 2015]．GoogLeNet 不写为GoogleNet，是为了向LeNet致敬．GoogLeNet赢得了 2014年ImageNet图像分类竞赛的冠军。"""
    r""" $ \qquad \qquad $ 网络结构：GoogLeNet 由9个 Inception v1 模块和 5 个池化层，以及其他一些卷积层和全连接层构成，总共为22层网络，如下图所示（清晰图见https://nndl.github.io/v/cnn-googlenet）"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片20.png', caption='图20 InceptionNet-V1的网络结构')
    """- ResNet """
    r""" $ \qquad $ 简介："""
    r""" $ \qquad \qquad $ 2015年，微软亚洲研究院何凯明等提出ResNet网络，以3.75%的top-5的错误率获得当时的ILSCRC大赛冠军。"""
    r""" $ \qquad \qquad $ 残差网络主要是通过残差块组成的，在提出残差网络之前，网络结构无法很深，在VGG中，卷积网络达到了19层，在GoogLeNet中，网络达到了22层。随着网络层数的增加，网络发生了退化（degradation）的现象：随着网络层数的增多，训练集loss逐渐下降，然后趋于饱和，当你再增加网络深度的话，训练集loss反而会增大。而引入残差块后，网络可以达到很深，网络的效果也随之变好。"""
    r""" $ \qquad $ 残差模块:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片21.png', caption='图21 残差模块')
    r""" $ \qquad $ 残差网络就是将很多个残差模块串联起来构成的一个非常深(网络深度可达100多层)的网络。"""

    """- 总结： """
    r"""$ \qquad $ 1.卷积神经网络是受生物学上感受野机制启发而提出的。[Hubel et al., 1959] 发现在猫的初级视觉皮层中存在两种细胞：简单细胞和复杂细胞。这两种细胞承担不同层次的视觉感知功能 [Hubel et al., 1962]。（1959年，David Hubel和Torsten Wiesel 也因此方面的贡献，于1981年获得诺贝尔生理学或医学奖。"""
    r"""$ \qquad $ 2. 受此启发，福岛邦彦（Kunihiko Fukushima）提出了一种带卷积和子采样操作的多层神经网络：神经认知机（Neocognitron）[Fukushima,1980]。(2021年美国鲍尔奖获得者)。但当时还没有反向传播算法，新知机采用了无监督学习的方式来训练。"""
    r"""$ \qquad $ 3. [LeCun et al., 1989]将反向传播算法引入了卷积神经网络，并在手写体数字识别上取得了很大的成功。"""
    r"""$ \qquad $ 4. AlexNet是第一个现代深度卷积网络模型，是深度学习技术在图像分类上真正突破的开端。在 AlexNet 之后，出现了很多优秀的卷积网络如VGGnet、InceptionNet系列、ResNet、DenseNet等。"""
    r"""$ \qquad $ 5. 目前，卷积神经网络已经成为计算机视觉领域的主流模型．通过引入跨层的直连边，可以训练上百层乃至上千层的卷积网络．随着网络层数的增加，卷积层越来越多地使用$1\times{1}$和$3\times{3}$大小的小卷积核，也出现了一些不规则的卷积操作，如空洞卷积、可变形卷积等。网络结构也逐渐趋向于全卷积网络，减少全连接层的作用。"""

with tab3:
    """- 当人的精神是聚焦在关心的那些事物上，便是注意力的体现，这种有意识的聚焦被称为**聚焦式注意力（Focus Attention）**。"""
    """- 无意识地，往往由外界刺激引发的注意力被称为**显著性注意力（Saliency-Based Attention）**。"""
    """- 但不论哪一种注意力，都是让人在某一时刻将注意力放到某些事物上，而忽略另外的一些事物，这就是**注意力机制（Attention Mechanism）**。"""
    """- 在深度学习领域，模型往往需要接收和处理大量的数据，然而在特定的某个时刻，往往只有少部分的某些数据是重要的，这种情况就非常需要Attention机制。"""
    """- 举例："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片22.jpg', caption='图22 机器翻译任务')
    r"""$ \qquad $ 传统的模型处理方式是seq-to-seq模型，其包含一个encoder端和一个decoder端，其中encoder端对"who are you"进行编码，然后将整句话的信息传递给decoder端，由decoder解码出"你是谁"。在这个过程中，decoder是逐字解码的，在每次解码的过程中，如果接收信息过多，可能会导致模型的内部混乱，从而导致错误结果的出现。"""
    r"""$ \qquad $ 若使用Attention机制，在生成"你"的时候和单词"you"关系比较大，和"who are"关系不大，所以我们更希望在这个过程中能够使用Attention机制，将更多注意力放到"you"上，而不要太多关注"who are"，从而提高整体模型的表现。"""
    r"""- Attention机制自提出以来，出现了很多不同Attention应用方式，但均是将模型的注意力聚焦在重要的事情上。"""
    """- 经典注意力机制：以机器翻译为例"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片23.jpg', caption='图23 机器翻译示例图')
    r"""$ \qquad $ 图23展示的是生成单词"machine"时的计算方式。首先将前一个时刻的输出状态$q_2$和Encoder的输出进行Attention计算，得到一个当前时刻的context，用公式可以这样组织："""
    st.latex(r"""[a_1,a_2,a_3,a_4]=softmax([s(q_2,h_1),s(q_2,h_2),s(q_2,h_3),s(q_2,h_4)])""")
    st.latex(r"""context=\sum_{i=1}^4{a_i\cdot{h_i}}""")
    r"""$ \qquad s(q_i,h_i)$ 表示注意力打分函数。context可以解释为：截止到当前已经有了"I love"，在此基础上下一个时刻应该更加关注源中文语句的那些内容。这就是关于Attention机制的一个完整计算。 """
    r"""$ \qquad $ 最后，将这个context和上个时刻的输出"love"进行融合作为当前时刻RNN单元的输入。"""
    """- 注意力打分函数："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片24.webp', caption='图24 Attention机制图')
    r"""$ \qquad $ 假设现在我们要对一组输入$H=[h_1,h_2,...,h_n]$使用Attention机制计算重要的内容，这里往往需要一个查询向量q(这个向量往往和做的任务有关，比如图23中用到的$q_2$) ，然后通过一个打分函数计算查询向量q和每个输入$h_i$之间的相关性，得出一个分数。"""
    st.latex(r"""a_i=softmax(s(h_i,q))=\frac{exp(s(h_i,q))}{\sum_{j=1}^n{exp(s(h_j,q))}}""")
    r"""$ \qquad $ 打分函数的计算方式："""
    r"""$ \qquad \qquad$ 1.加性模型：$s(h,q)=v^Ttanh(Wh+Uq)$"""
    r"""$ \qquad \qquad$ 2.点积模型：$s(h,q)=h^Tq$"""
    r"""$ \qquad \qquad$ 3.缩放点积模型：$s(h,q)=\frac{h^Tq}{\sqrt{D}}$"""
    r"""$ \qquad \qquad$ 4.双线性模型：$s(h,q)=h^TWq$"""