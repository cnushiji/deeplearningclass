import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tab1, tab2, tab3 = st.tabs(['1.Flow-based模型的不同之处', '2.Flow-based Model的建模思维', '3.Flow-based Model的理论推导&架构设计'])

with tab1:
    """- 流模型，它是一种比较独特的生成模型——它选择直接直面生成模型的概率计算，也就是把分布转换的积分式$ p_G(x)=\int_z{p(x|z)p(z)dz} $给硬算出来。要知道现阶段其他较火的生成模型，要么采用优化上界或采用对抗训练的方式去避开概率计算，从而寻找近似逼近真实分布的方法，但是流模型选择了一条硬路（主要是通过变换Jacobian行列式）来求解，在后文会详细介绍。"""
    """- 流模型有一个非常与众不同的特点是，它的转换通常是可逆的。也就是说，流模型不仅能找到从A分布变化到B分布的网络通路，并且该通路也能让B变化到A，简言之流模型找到的是一条A、B分布间的双工通路。当然，这样的可逆性是具有代价的——A、B的数据维度必须是一致的。"""
    """- A、B分布间的转换并不是轻易能做到的，流模型为实现这一点经历了三个步骤：最初的NICE实现了从A分布到高斯分布的可逆求解；后来RealNVP实现了从A分布到条件非高斯分布的可逆求解；GLOW实现了从A分布到B分布的可逆求解，其中B分布可以是与A分布同样复杂的分布，这意味着给定两堆图片，GLOW能够实现这两堆图片间的任意转换。"""

with tab2:
    """- 首先来回顾一下生成模型要解决的问题："""
    _, col1,_ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片43.jpg', caption='图43')
    """$ \qquad $ 如上图所示，给定两组数据z和x，其中z服从已知的简单先验分布π(z)（通常是高斯分布），x服从复杂的分布p(x)（即训练数据代表的分布），现在我们想要找到一个变换函数f，它能建立一种z到x的映射，使得每对于π(z)中的一个采样点，都能在p(x)中有一个（新）样本点与之对应。"""
    """$ \qquad $ 如果这个变换函数能找到的话，那么我们就实现了一个生成模型的构造。因为，p(x)中的每一个样本点都代表一张具体的图片，如果我们希望机器画出新图片的话，只需要从π(z)中随机采样一个点，然后通过，得到新样本点x，也就是对应的生成的具体图片。"""
    """$ \qquad $ 所以，接下来的关键在于，这个变换函数f如何找呢？我们先来看一个最简单的例子。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片44.jpg', caption='图44')
    """$ \qquad $ 如上图所示，假设z和x都是一维分布，其中z满足简单的均匀分布：$ \pi{(z)}=1(z\in{[0,1]}) $，x也满足简单均匀分布：$ p(x)=0.5(x\in{[1,3]}) $。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片45.jpg', caption='图45')
    """$ \qquad $ 那么构建z与x之间的变换关系只需要构造一个线性函数即可：$x=f(z)=2z+1$。"""
    """$ \qquad $ 下面再考虑非均匀分布的更复杂的情况:"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片46.jpg', caption='图46')
    """$ \qquad $ 如上图所示，π(z)与p(x)都是较为复杂的分布，为了实现二者的转化，我们可以考虑在很短的间隔上将二者视为简单均匀分布，然后应用前边方法计算小段上的，最后将每个小段变换累加起来（每个小段实际对应一个采样样本）就得到最终的完整变换式f。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片47.jpg', caption='图47')
    """$ \qquad $ 如上图所示，假设在$[𝑧^{′},𝑧^{′}+\Delta{z}]$上$\pi{(z)}$近似服从均匀分布，在$[x^{′},x^{′}+\Delta{x}]$上p(x)也近似服从均匀分布，于是有$𝑝(𝑥^{′})\Delta{𝑥}=\pi{(𝑧^{′})}\Delta{𝑧}$（因为变换前后的面积/即采样概率是一致的），当$\Delta{x}$与$\Delta{𝑧}$极小时，有："""
    st.latex(r"""p(x^{'})=\pi{(z)}\frac{dz}{dx}""")
    """$ \qquad $ 又考虑到$\\frac{dz}{dx}$有可能是负值（如下图所示），而$p(x^{'})$与$\pi{(z^{'})}$都为非负，所以$p(x^{'})$与$\pi{(z^{'})}$的实际关系为："""
    st.latex(r"""p(x^{'})=\pi{(z^{'})}|\frac{dz}{dx}|""")
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片48.jpg', caption='图48')
    """$ \qquad $ 下面进一步地做推广，我们考虑z与x都是二维分布的情形。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片49.jpg', caption='图49')
    """$ \qquad $  如上图所示，z与x都是二维分布，左图中浅蓝色区域表示初始点在方向上移动$\Delta$，在方向上移动$\Delta$所形成的区域，这一区域通过映射，形成右图所示x域上的浅绿色菱形区域。其中，二维分布$\pi{(z)}$与$p(x)$均服从简单均匀分布，其高度在图中未画出（垂直纸面向外）。"""
    """$ \qquad $ 因为蓝色区域与绿色区域具有相同的体积，所以有："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片50.png', caption='图50')
    """$ \qquad $ 其中"""
    st.latex(r"""det\left[
                        \begin{array}{cc}
                          \Delta{x_{11}} & \Delta{x_{21}} \\
                          \Delta{x_{12}} & \Delta{x_{22}} 
                        \end{array}
                    \right] """)
    r"""代表行列式计算，它的计算结果等于上图中浅绿色区域的面积（行列式的定义）。下面我们将$\Delta{z_1},\Delta{z_2}$移至左侧，得到："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片51.png', caption='图51')
    """$ \qquad $ 即："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片52.png', caption='图52')
    """$ \qquad $  在$\Delta{z_1},\Delta{z_2}$很小时，有："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片53.png', caption='图53')
    """$ \qquad $ 即："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片54.png', caption='图54')
    """$ \qquad $ 其中$J_f$表示f运算的雅各比行列式，根据雅各比行列式的逆运算，我们得到:"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片55.png', caption='图55')
    """$ \qquad $ 其中$f^{-1}$代表从x变换为z的变换式，即：$z=f^{-1}(x)$。"""
    """$ \qquad $ 至此，我们得到了一个比较重要的结论：如果z与x分别满足两种分布，并且z通过函数f能够转变为x，那么z与x中的任意一组对应采样点$z^{'}$与$x^{'}$之间的关系为："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片56.png', caption='图56')
    """$ \qquad $ 那么基于这一结论，再带回到生成模型要解决的问题当中，我们就得到了Flow-based Model（流模型）的初步建模思维。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片57.jpg', caption='图57')
    """$ \qquad $  如上图所示，为了实现$z\sim{\pi{(z)}}$到$x=G(z)\sim{p_G{(x)}}$间的转化，待求解的生成器G的表达式为："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片58.png', caption='图58')
    """$ \qquad $ 基于前面推导，我们有$p_G{(x)}$中的样本点与$\pi{(z)}$中的样本点间的关系为："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片59.png', caption='图59')
    """$ \qquad $  其中$ z^i=G^{-1}{(x^i)} $。"""
    """$ \qquad $ 所以，如果$G^{*}$的目标式能够通过上述关系式求解出来，那么我们就实现了一个完整的生成模型的求解。Flow-based Model就是基于这一思维进行理论推导和模型构建，下面将会详细解释Flow-based Model的求解过程。"""
with tab3:
    """$ \qquad $ 我们关注一下上一章中引出的式子："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片60.png', caption='图60')
    """$ \qquad $ 将其取log，得到："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片61.png', caption='图61')
    """$ \qquad $ 现在，如果想直接求解这个式子有两方面的困难。第一个困难是，$det(J_{G^{-1}})$是不好计算的——由于$G^{-1}$的Jacobian矩阵一般维度不低（譬如$256\\times{256}$矩阵），其行列式的计算量是异常巨大的，所以在实际计算中，我们必须对$G^{-1}$的Jacobian行列式做一定优化，使其能够在计算上变得简洁高效。第二个困难是，表达式中出现了$G^{-1}$，这意味着我们要知道$G^{-1}$长什么样子，而我们的目标是求G，所以这需要巧妙地设计G的结构使得$G^{-1}$也是好计算的。"""
    """$ \qquad $ 下面我们来逐步设计G的结构，首先从最基本的架构开始构思。考虑到$G^{-1}$必须是存在的且能被算出，这意味着G的输入和输出的维度必须是一致的并且G的行列式不能为0。然后，既然$G^{-1}$可以计算出来，而$log_{p_G(x^i)}$的目标表达式只与$G^{-1}$有关，所以在实际训练中我们可以训练$G^{-1}$对应的网络，然后想办法算出G来并且在测试时改用G做图像生成。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片62.jpg', caption='图62')
    """$ \qquad $ 如上图所示，在训练时我们从真实分布$p_{data}{(x)}$中采样出$x^i$，然后去训练$G^{-1}$，使得通过$G^{-1}$生成的$z^i=G^{-1}(x^i)$满足特定的先验分布；接下来在测试时，我们从z中采样出一个点$z^j$，然后通过G生成的样本$x^j=G(z^j)$就是新的生成图像。"""
    """$ \qquad $ 接下来开始具体考虑G的内部设计，为了让$G^{-1}$可以计算并且G的Jacobian行列式也易于计算，Flow-based Model采用了一种称为耦合层（Coupling Layer）的设计来实现。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片63.jpg', caption='图63')
    """$ \qquad $ 如上图所示，z和x都会被拆分成两个部分，分别是前$1\sim{d}$维和后$d+1\sim{D}$维。从z变化为x的计算式为：z的$1\sim{d}$维直接复制（copy）给x的$1\sim{d}$维；z的$d+1\sim{D}$维分别通过F和H两个函数变换为$\\beta_{d+1,...,D}$和$\\gamma_{d+1,...,D^{'}}$，然后通过$x_i=\\beta_iz_i+\gamma_i(i=d+1,...,D)$的仿射计算（affine）传递给x。综上，由z传给x的计算式可以写为："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片64.png', caption='图64')
    """$ \qquad $ 其逆运算的计算式，即由x传给z的计算式，可以非常方便地推导出来为："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片65.png', caption='图65')
    """$ \qquad $ 上面我们说明了，这样设计的耦合层能快速计算出$G^{-1}$，下面我们来说明，其在G的Jacobian行列式的计算上也是非常简便。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片66.jpg', caption='图66')
    """$ \qquad $ 上图展示了G的Jacobian行列式的计算矩阵。首先由于$z_{1,...,d}$直接传递给$x_{1,...,d}$所以Jacobian矩阵的左上角区域是单位矩阵I，然后$x_{1,...,d}$完全不受$z_{d+1,...,D}$影响，所以Jacobian矩阵的右上角区域是零矩阵O，这导致Jacobian矩阵的左下角区域的值对Jacobian矩阵行列式的计算没有影响，也就无需考虑。最后我们关注Jacobian矩阵的右下角区域，由于$x_i=\\beta_iz_i+\gamma_i(i>d)$，所以只有在$i=j$的情况下$\\frac{\partial{x_i}}{\partial{z_j}}\\ne{0}$，而在处，所以Jacobian矩阵的右下角区域是一个对角矩阵。"""
    """$ \qquad $ 最终，该G的Jacobian的行列式计算式就表示为："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片67.png', caption='图67')
    """$ \qquad $ 这确实是一个易于计算的简单表达式。接下来可以考虑，由于上述措施对G做了诸多限制，导致G的变换能力有限，所以我们可以堆叠多个G，去增强模型的变换拟合能力。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片68.jpg', caption='图68')
    """$ \qquad $ 如上图所示，我们将多个耦合层堆叠在一起，从而形成一个更完整的生成器。但是这样会有一个新问题，就是最终生成数据的前d维与初始数据的前d维是一致的，这会导致生成数据中总有一片区域看起来像是固定的图样（实际上它代表着来自初始高斯噪音的一个部分），我们可以通过将复制模块（copy）与仿射模块（affine）交换顺序的方式去解决这一问题。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片69.jpg', caption='图69')
    """$ \qquad $ 如上图所示，通过将某些耦合层的copy与affine模块进行位置上的互换，使得每一部分数据都能走向copy->affine->copy->affine的交替变换通道，这样最终的生成图像就不会包含完全copy自初始图像的部分。值得说明的是，在图像生成当中，这种copy与affine模块互换的方式有很多种，下面举两个例子来说明："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片70.jpg', caption='图70')
    """$ \qquad $ 上图展示了两种按照不同的数据划分方式做copy与affine的交替变换。左图代表的是在像素维度上做划分，即将横纵坐标之和为偶数的划分为一类，和为奇数的划分为另外一类，然后两类分别交替做copy和affine变换（两两交替）；右图代表的是在通道维度上做划分，通常图像会有三通道，那么在每一次耦合变换中按顺序选择一个通道做copy，其他通道做affine（三个轮换交替），从而最终变换出我们需要的生成图形出来。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片71.jpg', caption='图71')
    """$ \qquad $  更进一步地，如何进行copy和affine的变换能够让生成模型学习地更好，这是一个可以由机器来学习的部分，所以我们引入W矩阵，帮我们决定按什么样的顺序做copy和affine变换，这种方法叫做1×1 convolution（被用于知名的GLOW当中）。1×1 convolution只需要让机器决定在每次仿射计算前对图片哪些区域实行像素对调，而保持copy和affine模块的顺序不变，这实际上和对调copy和affine模块顺序产生的效果是一致的。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片72.jpg', caption='图72')
    """$ \qquad $  这种对调的原理非常简单，如上图所示举例，假设我们需要将（3,1,2）向量替换成（1,2,3）向量，只需要将w矩阵定义为图中所示矩阵即可。下面我们看一下，将w引入flow模型之后，对于原始的Jacobian行列式的计算是否会有影响。"""
    """$ \qquad $ 对于每一个$3\\times{3}$维划分上的仿射操作来说，由$x=f(z)=Wz$我们可以得到f的Jacobian行列式的计算结果为："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片73.png', caption='图73')
    """$ \qquad $ 代入到整个含有d个$3\\times{3}$维的仿射变换矩阵当中，得到最终的Jacobian行列式的计算结果就为：$(det(W))^{d\\times{d}}$，如下图所示："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片74.jpg', caption='图74')
    """$ \qquad $ 因此，引入1×1 convolution后的G的Jacobian行列式计算依然非常简单，所以引入1×1 convolution是可取的，这也是GLOW这篇Paper最有突破和创意的地方。"""
    _, col1 = st.columns([1, 4])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片75.jpg', caption='图75')
    """$ \qquad $ 综上，关于Flow-based Model的理论讲解和架构分析就全部结束了，它通过巧妙地构造仿射变换的方式实现不同分布间的拟合，并实现了可逆计算和简化雅各比行列式计算的功能和优点，最终我们可以通过堆叠多个这样的耦合层去拟合更复杂的分布变化（如上图所示），从而达到生成模型需要的效果。"""
