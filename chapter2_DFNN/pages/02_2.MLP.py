import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['1.网络架构', '2.优化', '3.反向传播(BP)', '4.初始化','5.正则化','6.示例——mnist'])

with tab1:
    """- 人工神经元模型："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片10.png', caption='图10')
    """- 激活函数："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片11.png', caption='图11')
    """- MLP："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片12.png', caption='图12')
    """- 万有逼近定理："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片13.png', caption='图13')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片14.png', caption='图14')

with tab2:
    """- 优化的原因："""
    r"""$ \qquad $ 人工智能的诸多应用需要用大量数据进行学习。通过学习，神经网络可以自动地从数据中提取特征，并根据学习的特征自动决策。"""
    r"""$ \qquad $ 学习(learning)在数学上表达为对一个损失函数𝐿(𝑥,𝜃)的优化问题。"""
    r"""$ \qquad $ 为了求解此优化问题，一类重要的优化方法是基于梯度下降的优化算法。"""
    """- 优化的定义："""
    r"""$ \qquad $ 数学中的优化：通过改变自变量x以最小化或最大化函数𝑓(𝑥)，𝑓(𝑥)称为目标函数或者损失函数。"""
    r"""$ \qquad $ 人工智能算法中的优化：训练数据集X，样本标签Y，神经网络输出𝑓(𝑥,𝜃),𝑝(𝑥,𝑦)是联合分布，因此损失函数定义为"""
    st.latex(r"""L(\theta)=\frac{1}{m}\sum_i{l(f(x^{(i)},\theta),y^{(i)})}""")
    r"""$ \qquad \qquad $ 其中，𝑙是每个样本的损失函数。优化的目标是找到最优的参数$ \theta_∗ $使得损失函数L取得最小值。"""

    """- 梯度下降："""
    r"""$ \qquad $ 梯度："""
    st.latex(r"""\nabla{f(x)}=[\frac{\partial{f}}{\partial{x_1}},...,\frac{\partial{f}}{\partial{x_n}}]^T""")
    r"""$\qquad \qquad $ 几何含义：$\nabla{f(x)}$函数增长最快的方向。"""
    r"""$\qquad \qquad $ 直观启发：在负梯度方向上函数值下降最快。因此，如果我们每次沿着负梯度方向移动$ \theta $，那么函数值就会不断下降。"""
    _, col1, _ = st.columns([2, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片15.png', caption='图15')

    r"""$ \qquad $ 公式："""
    st.latex(r"""\theta_{n+1}=\theta_n-\alpha{\frac{1}{m}\sum_i{\nabla_{\theta_n}{l(x^{(i)},\theta_n)}}}""")
    r"""$ \qquad \qquad $ 其中$ \alpha $是学习率，表示每一步更新的步长。"""
    r"""$ \qquad $ 示例：在[−1,1]上均匀采样的m个样本数据，通过一个隐藏层的神经网络学习函数$ y=0.5𝑥^2 $。 损失函数定义如下"""
    st.latex(r"""min{L(x,\theta)}=\frac{1}{m}\sum_i{(f(x^{(i)},\theta),y^{(i)})^2}""")
    r"""$ \qquad $ 优化结果："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片16.jpg', caption='图16')

    """- 深度学习中的优化方法："""
    r"""$ \qquad $ 1.随机梯度下降——Vanilla-SGD："""
    r"""$ \qquad \qquad $ 问题：数据集有几十上百万条数据时，用整个数据集训练，计算代价高，所需计算内存大。"""
    r"""$ \qquad \qquad $ 如何解决? → 随机梯度优化(SGD)：每一次更新参数时只用一个样本，从而使得计算可行。“以时间换资源”"""
    r"""$ \qquad \qquad $ 公式："""
    st.latex(r"""\theta_{n+1}=\theta_n-\alpha{\cdot{\nabla_{\theta_n}l(x^{(i)},\theta_n)}}""")
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片17.png', caption='图17')
    r"""$ \qquad \qquad $ """
    r"""$ \qquad $ 2.随机梯度下降——minibatch-SGD："""
    r"""$ \qquad \qquad $ 问题：原始SGD方法收敛速度慢，训练不稳定。"""
    r"""$ \qquad \qquad $ 如何改进？→ minibatch SGD: 在每一次更新参数时用一组样本，而不是只用一个样本。"""
    r"""$ \qquad \qquad $ 公式："""
    st.latex(r"""g_n=\frac{1}{|B|}\sum_{i\in{B}}{\nabla_{\theta_n}l(x^{(i)},\theta_n)},\theta_{n+1}=\theta_n-\alpha{\cdot{g_n}}""")
    r"""$ \qquad \qquad \qquad $ 其中，B是一个minibatch，$|𝐵|$表示minibatch的大小。"""
    r"""$ \qquad $ 3.动量方法——Momentum-SGD："""
    r"""$ \qquad \qquad $ 动量方法(Momentum-SGD)是利用历史梯度信息和当前梯度信息给出当梯度下降方向，加速SGD在正确方向的下降并抑制震荡。"""
    r"""$ \qquad \qquad $ 公式："""
    st.latex(r"""g_t=\frac{1}{|B|}\sum_{i\in{B}}{\nabla_{\theta_t}l(x^{(i)},\theta_t)}""")
    st.latex(r"""m_t=\gamma{m_{t-1}}+(1-\gamma)g_t""")
    st.latex(r"""\theta_{t+1}=\theta_t-m_t""")
    col1, col2 = st.columns([1,1.1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片18.png', caption='图18 SGD')
    with col2:
        st.image('./chapter2_DFNN/pages/图片/图片19.png', caption='图19 Momentum')
    r"""$ \qquad $ 4.Nestrov加速梯度(NAG)："""
    r"""$ \qquad \qquad $ 直观想法：在目标函数有增高趋势之前，减缓更新速率。"""
    r"""$ \qquad \qquad $ NAG原理：计算未来位置的梯度，并结合历史梯度信息得到下降方向。"""
    r"""$ \qquad \qquad $ 公式："""
    st.latex(r""" g_t=\frac{1}{|B|}\sum_{i\in{B}}{\nabla_{\theta_t}l(x^{(i)},\theta_t-\gamma{m_{t-1}})} """)
    st.latex(r"""m_t=\gamma{m_{t-1}}+(1-\gamma)g_t""")
    st.latex(r"""\theta_{t+1}=\theta_t-m_t""")
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片20.png', caption='图20')
    r"""$ \qquad \qquad $ """
    r"""$ \qquad $ 5.自适应学习率方法(AdaGrad/RMSProp)"""
    r"""$ \qquad \qquad $ 自适应调整学习率：随着优化的进行，参数梯度的变化越来越小，而一直用固定的学习率是否合适？"""
    r"""$ \qquad \qquad $ AdaGrad：随着学习过程的进行，动态调整学习率。"""
    r"""$ \qquad \qquad \qquad $ 公式："""
    st.latex(r"""V_t=\sum_{i=1}^t{g_i^2}, \qquad \theta_{t+1}=\theta_t-\eta{\cdot{\frac{1}{\sqrt{V_t+\varepsilon}}g_t}}""")
    r"""$ \qquad \qquad $ RMSProp：修改AdaGrad的梯度平方和累加为指数加权的移动平均。"""
    r"""$ \qquad \qquad \qquad $ 公式："""
    st.latex(r"""g_t=\nabla_{\theta_t}{L(B,\theta_t)}""")
    st.latex(r"""E[g_t^2]_t=\beta{E[g_t^2]_{t-1}}+(1-\beta)g_t^2""")
    st.latex(r"""\theta_{t+1}=\theta_t-\eta{\frac{1}{\sqrt{E[g_t^2]_t+\varepsilon}}g_t}""")
    r"""$ \qquad $ 6.Adam优化方法："""
    r"""$ \qquad \qquad $ 结合一阶动量和二阶动量信息。"""
    r"""$ \qquad \qquad $ 直观理解：“Momentum方法 + RMSProp方法”"""
    r"""$ \qquad \qquad $ 公式："""
    st.latex(r"""g_t=\nabla_{\theta_t}{L(B,\theta_t)}""")
    st.latex(r"""m_t=\beta_1{m_{t-1}}+(1-\beta_1)g_t, \qquad \hat{m}_t=\frac{\hat{m}_t}{1-\beta_1^t}""")
    st.latex(r"""v_t=E[g_t^2]_t=\beta_2E[g_t^2]_{t-1}+(1-\beta_2)g_t^2, \qquad \hat{v}_t=\frac{\hat{v}_t}{1-\beta_2^t}""")
    st.latex(r"""\theta_{t+1}=\theta_t-\eta{\cdot{\frac{\hat{m}_t}{\sqrt{\hat{v}_t+\varepsilon}}}}""")
    r"""$ \qquad \qquad $ 做了一个偏置校正bias correction，防止在训练初始阶段过于偏向0。"""

    r"""$ \qquad $ 不同优化方法的比较示例："""
    r"""$ \qquad \qquad $ 示例，学习一个定义在[-1,1]上的一维函数$𝑦=0.5𝑥^2$。"""
    col1, col2= st.columns([1,1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片21.gif', caption='图21')
    with col2:
        st.image('./chapter2_DFNN/pages/图片/图片22.gif', caption='图22')
    _, col1, _= st.columns([1, 1.2, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片23.gif', caption='图23')
    r"""$ \qquad $ 总结："""
    r"""$ \qquad \qquad $ 1.回忆一下要点：损失函数+梯度下降，即利用当前梯度，历史梯度，下一步梯度的不同结合策略形成了不同的优化方法方法。"""
    r"""$ \qquad \qquad $ 2.哪种优化方法最好？没有一种优化方法对所有的问题都是最好的，通常推荐Adam方法。"""
    r"""$ \qquad \qquad $ 3.梯度如何计算？→反向传播算法BP"""

with tab3:
    """- 微积分中的链式法则："""
    r"""$ \qquad $ 微积分中的链式法则用于计算复合函数的导数。反向传播是一种计算链式法则的算法，使用高效的特定运算顺序。"""
    r"""$ \qquad $ 假设$x\in{R^m},y\in{R^n}$,𝑔是从$𝑅^𝑚$到$𝑅^𝑛$的映射，𝑓是从$𝑅^𝑛$到𝑅的映射。如果$𝑦=𝑔(𝑥)$并且$𝑧=𝑓(𝑦)$，那么"""
    st.latex(r"""\nabla_x{z}=(\frac{\partial{y}}{\partial{x}})^T\nabla_y{z}""")
    r"""$ \qquad $ 这里$\frac{\partial{y}}{\partial{x}}$是g的$n\times{m}$的Jacobian矩阵。"""
    r"""$ \qquad $ 张量的链式法则：如果$𝒀=𝑔(𝑿)$并且$𝑧=𝑓(𝒀)$，那么"""
    st.latex(r"""\nabla_x{z}=\sum_j{(\nabla_X{Y_j})\frac{\partial{z}}{\partial{Y_j}}}""")
    """- 递归地使用链式法则反向传播："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片24.png', caption='图24')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片25.png', caption='图25')
    r"""$ \qquad $ 前向过程："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片26.png', caption='图26')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片27.png', caption='图27')
    _, col1, _ = st.columns([1, 4, 1])
    r"""$ \qquad $ 反向过程："""
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片28.png', caption='图28')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片29.png', caption='图29')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片30.png', caption='图30')
    r"""$ \qquad \qquad $ 上图中，$\sigma'{(z)}$是常数，因为z在Forward Pass的时候就已经被确定了。"""
    r"""$ \qquad $ 最后的问题只有如何计算$ \frac{\partial{C}}{\partial{z'}},\frac{\partial{C}}{\partial{z"}} $，如下图："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片31.png', caption='图31')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片32.png', caption='图32')
    r"""$ \qquad $ 从output layer计算$ \frac{\partial{C}}{\partial{z}} $，如下图："""
    st.latex(r"""\frac{\partial{C}}{\partial{z_3}}=\sigma'{z_3}[w_5\frac{\partial{C}}{\partial{z_5}}+w_6\frac{\partial{C}}{\partial{z_6}}]""")
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片33.png', caption='图33')
    r"""$ \qquad $ 总结："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片34.png', caption='图34')

    """- 梯度消失和梯度爆炸："""
    r"""$ \qquad $ Vanishing gradient problem: 由于Sigmoid型函数的饱和性，饱和区的导数更是接近于0。这样，误差经过每一层传递都会不断衰减．当网络层数很深时，梯度就会不停衰减，甚至消失，使得整个网络很难训练．这就是所谓的梯度消失问题（Vanishing Gradient Problem），也称为梯度弥散问题。"""
    r"""$ \qquad $ Gradient exploding problem: 当深层神经网络或循环卷积神经网络中网络权值初始化值太大时，随着网络深度的加深，梯度中的重复相乘项会导致梯度呈指数级增长，梯度变的非常大，从而导致网络权值的大幅更新，并因此导致网络的不稳定。"""
    r"""$ \qquad $ 如何解决？梯度剪切、ReLU激活、正则化、BN、残差连接等。"""

    """- 在pytorch中如何计算导数？"""
    r"""$ \qquad $ Pytorch的反向传播时计算梯度默认采用累加机制，即当未手动对上一次计算的梯度进行清零时，当前梯度值是本次梯度值加上上一次的梯度值。"""
    st.code(r"""Optimizer.zero_grad() 或
w.grad.data.zero_() 或
b.grad.data.zero_()
    """)
    r"""$ \qquad $ 反向传播的中间缓存会被清空，如果需要进行多次反向传播，需要制定backward中的参数"""
    st.code(r"""retain_graph=True""")
    r"""$ \qquad $ 这在对非标量函数求导时有用。"""
    r"""$ \qquad $ 计算图："""
    r"""$ \qquad \qquad $ 为了更精确地描述反向传播算法，使用更精确的计算图(computational graph)语言是很有帮助的。计算图是数学运算的图形化表示。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片35.png', caption='图35')
    r"""$ \qquad \qquad $ 图中的每一个节点来表示一个变量。变量可以是标量、向量、矩阵、张量或者甚至是另一类型的变量。边表示运算或操作operation。"""
    r"""$ \qquad \qquad $ 计算图中每个叶子节点为一个输入变量或常量。每个非叶子节点表示一个中间变量，是一个或多个变量的简单函数。"""
    r"""$ \qquad \qquad $ 根据计算图搭建方式的不同，可将计算图分为静态图和动态图。"""
    r"""$ \qquad $ 自动微分Auto-differentiation："""
    r"""$ \qquad \qquad $ 前向模式： 按计算图中计算方向的相同方向来递归地计算梯度。"""
    r"""$ \qquad \qquad $ 反向模式：按计算图中计算方向的相反方向来递归地计算梯度。"""
    r"""$ \qquad \qquad $ 无论是前向模式还是反向模式，二者计算梯度的方式都相同。在前馈神经网络的参数学习中，风险函数$𝑓:𝑅^𝑁→𝑅$，输出为标量，因此采用反向模式是最有效的计算方式。"""
    r"""$ \qquad $ 静态计算图和动态计算图："""
    r"""$ \qquad \qquad $ 静态计算图是在编译时构建计算图，计算图构建好之后在程序运行时不能改变。而动态计算图是在程序运行时动态构建。"""
    r"""$ \qquad \qquad $ 目前主流的深度学习框架里，Theano和Tensorflow1.x版本采用的是静态计算图，而Pytorch采用的是动态计算图。Tensorflow2.0也支持动态计算图。"""
    r"""$ \qquad \qquad $ 静态计算图可以在构建时进行优化，并行能力强，但灵活性比较差。动态计算图不容易优化，当不同输入的网络结构不一致时，难以并行计算，灵活性比较高。（爬梯度，边爬边搭，和边搭边爬）"""

with tab4:
    """- 为什么进行初始化？"""
    r"""$ \qquad $ 权重初始化是赋给网络一个好的初值，目的是在通过深层神经网络的前馈路径中，防止层激活输出爆炸或者消失。"""
    """- Xavier初始化："""
    r"""$ \qquad $ 2010论文《Understanding the difficulty of training deep feedforward neural networks》"""
    r"""$ \qquad $ Motivation：为了使得网络中信息更好的流动，每一层输出的方差应该尽量相等。"""
    r"""$ \qquad $ 1.均匀分布："""
    r"""$ \qquad \qquad $ 用法： """
    st.code("""torch.nn.init.xavier_uniform_(tensor,gain=1)""")
    r"""$ \qquad \qquad $ Xavier初始化方法中服从均匀分布U(-a,a),分布的参数如下，"""
    st.code("""a=gain*sqrt(6/fan_in+fan_out)""")
    r"""$\qquad \qquad $ 这里有一个gain, 增益的大小是依据激活函数类型来设定。"""
    r"""$ \qquad \qquad $ 例如，"""
    st.code("""nn.init.xavier_uniform_(w,gain=nn.nint.calculate_gain(‘relu’))""")
    r"""$ \qquad $ 2.正态分布："""
    r"""$\qquad \qquad $ Xavier初始化方法中权重服从正态分布:"""
    st.code(r"""W~N(mean,std^2)
mean=0
std=gain*sqrt(2/fan_in+fan_out)
""")
    r"""$ \qquad \qquad $ 用法："""
    st.code(r"""torch.nn.init.xavier_normal_(tensor,gain=1)""")
    """- Kaiming初始化："""
    r"""$ \qquad $ xavier初始化方法在tanh等激活函数上表现的很好，而在relu这一类激活函数表现不佳，kaiming初始化方法是对此进行的改进。"""
    r"""$ \qquad $ Motivation: 在ReLU网络中，假定每一层有一半的神经元被激活，另一半为0。所以，要保持方差不变，只需要在Xavier的基础上再除以2即可。"""
    r"""$ \qquad $ 1.均匀分布："""
    st.code(r"""W~U(-bound,bound),bound=sqrt(6/(1+a^2)*fan_in)""")
    r"""$ \qquad \qquad $ 用法："""
    st.code(r"""torch.nn.init.kaiming_uniform_(tensor,a=0,mode=‘fan_in’,nonlinearity=‘leaky_relu’)""")
    r"""$ \qquad \qquad $ 其中，a为激活函数负半轴的斜率，relu=0。"""
    r"""$ \qquad \qquad $ Mode: 可选为fan_in或fan_out。fan_in使正向传播时，方差一致；fan_out使反向传播时，方差一致。"""
    r"""$ \qquad \qquad $ Nonlinearity: 可选relu和leaky_relu,默认值为leak_relu。"""
    r"""$ \qquad $ 2.正态分布："""
    st.code(r"""torch.nn.init.kaiming_normal_(tensor,a=0,mode=‘fan_in’,nonlinearity=‘leaky_relu’)""")
    r"""$ \qquad \qquad $ 此为0均值的正太分布，即N~(0,std),且"""
    st.code(r"""std=sqrt(2/(1+a^2)*fan_in)""")
    r"""$ \qquad \qquad $ 其中，a为激活函数的负半轴的斜率，relu是0。"""
    r"""$ \qquad \qquad $ mode：可选为fan_in或fan_out,fan_in使正向传播时，方差一致；fan_out使反向传播时，方差一致。"""
    r"""$ \qquad \qquad $ nonlinearity：可选relu或leaky_relu,默认值为leaky_relu。"""
    """- 其他初始化方式："""
    r"""$ \qquad $ 1.均匀分布初始化："""
    st.code(r"""torch.nn.init.uniform_(tensor,a=0,b=1)""")
    r"""$ \qquad \qquad $ 使值服从均与分布U(a,b)。"""
    r"""$ \qquad $ 2.Gauss分布初始化："""
    st.code(r"""torch.nn.init.normal_(tensor,mean=0,std=1)""")
    r"""$ \qquad \qquad $ 使值服从正太分布N(mean,std),默认值为0、1。"""
    r"""$ \qquad $ 3.常数初始化："""
    st.code(r"""torch.nn.init.eye_(tensor)""")
    r"""$ \qquad \qquad $ 使值为常数val nn,init,constant_(w,0.3)。"""
    r"""$ \qquad $ 4.单位矩阵初始化："""
    st.code(r"""torch.nn.init.constant_(tensor,val)""")
    r"""$ \qquad \qquad $ 将二维tensor初始化为单位矩阵(the identity matrix)"""
    r"""$ \qquad $ 5.正交初始化："""
    st.code(r"""torch.nn.init.orthogonal_(tensor,gain=1)""")
    r"""$ \qquad \qquad $ 使得tensor是正交的,正交初始化可以使得卷积核更加紧凑，可以去除相关性，使模型更容易学到有效的参数。"""
    r"""$ \qquad $ 6.稀疏初始化："""
    st.code(r"""torch.nn.init.sparse_(tensor,sparsity,std=0.01)""")
    r"""$ \qquad \qquad $ 从正太分布N~(0,std)中进行稀疏化，使每一个column有一部分为0，Sparsity是每一个column稀疏的比例，即为0的比例。"""
    r"""$ \qquad $ 7.计算增益："""
    st.code(r"""torh.nn.init.calculate_gain(nonlinearity,param=None)""")
    _, col1, _ = st.columns([1, 1, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片36.png', caption='图36')

with tab5:
    """- $L_1,L_2$正则化和权重衰减："""
    r"""$ \qquad $ 正则化定义：为减少测试误差而设计的显式训练策略。正则化是对学习算法的修改——旨在减少泛化误差而不是训练误差。"""
    r"""$ \qquad $参数范数惩罚：许多正则化方法通过对目标函数 J 添加一个参数范数惩罚 Ω(θ)，限制模型的学习能力。我们将正则化后的目标函数记为："""
    st.latex(r"""\tilde{J}(\theta;X,y)=J(\theta;X,y)+\alpha{\Omega{(\theta)}}""")
    r"""$ \qquad $ 在神经网络中我们通常只对每一层仿射变换的权重做惩罚而不对偏置做正则惩罚。每个权重会指定两个变量如何相互作用；每个偏置仅控制一个单变量。这意味着，我们不对其进行正则化也不会导致太大的方差。"""
    r"""$ \qquad $ $L_1$参数正则化："""
    st.latex(r"""\Omega{(\theta)}=\Vert \omega \Vert_1=\sum_i{|\omega_i|}""")
    r"""$ \qquad $ $L_2$参数正则化(也被称为权重衰减/岭回归/Tikhonov正则)："""
    st.latex(r"""\Omega{(\theta)}=\frac{1}{2}\Vert \omega \Vert_2^2""")
    r"""$ \qquad $ 通过向目标函数添加一个正则项，使权重更加接近原点。"""
    r"""$ \qquad $ L2正则化的作用在于削减权重，降低模型过拟合，其行为直接导致每轮迭代过程中的权重weight参数被削减/惩罚一部分，故也称为权重衰减。"""
    r"""$ \qquad $ 从这个角度看，不论用L1正则化还是L2正则化，亦或是其它正则化方法，只要是削减了权重，那都可以称为weight decay。"""
    r"""$ \qquad $ L1,L2正则化减少权重使得网络对丢失特定神经元连接的鲁棒性提高。"""

    """- Batch Normalization："""
    r"""$ \qquad $ 随着训练的进行，网络中的参数也随着梯度下降再不断更新。"""
    r"""$ \qquad \qquad $ 一方面，当底层网络中参数发生微弱变化时，由于每一层中的线性变换与非线性激活映射，这些微弱变换随着网络层数的加深而被放大（类似蝴蝶效应）。"""
    r"""$ \qquad \qquad $ 另一方面，参数的变化导致每层的输入分布会发生改变，进而上层的网络需要不停地去适应这些分布变化，使得我们的模型训练变得困难。上述这一现象叫做Internal Covariate Shift。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片37.png', caption='图37')
    r"""$ \qquad $ 1.什么是Internal Covariate Shift?"""
    r"""$ \qquad \qquad $ 定义：在深层网络训练的过程中，由于网络中参数变化而引起内部节点数据分布发生变化的这一过程被称作Internal Covariate Shift。"""
    r"""$ \qquad \qquad $ 解释：我们定义每一层的线性变换为$ Z^{[l]}=W^{[l]}\times{input}+b^{[l]} $，其中l代表层数；非线性变换为$ A^{[l]}=g^{[l]}(Z^{[l]}) $,其中$g^{[l]}(\cdot)$为第l层的激活函数。随着梯度下降的进行，每一层的参数$W^{[l]}$与$b^{[l]}$都会被更新，那么$Z^{[l]}$的分布也就发生了改变，进而𝐴^([𝑙]) 也同样出现分布的改变。而$A^{[l]}$作为第l+1层的输入，意味着l+1层就需要去不停地适应这种数据分布的变化，这一过程就被叫做Internal Covariate Shift。
"""
    r"""$ \qquad $2.Internal Covariate Shift会带来什么问题？"""
    r"""$ \qquad \qquad $ 上层网络需要不停地调整来适应输入数据分布的变化，导致网络学习速率的降低。"""
    r"""$ \qquad \qquad $ 网络的训练过程容易陷入梯度饱和区，减缓网络收敛速度。"""
    r"""$ \qquad \qquad $对于激活函数梯度饱和问题，有两种解决思路："""
    r"""$ \qquad \qquad \qquad $ 第一种就是更为非饱和性激活函数，例如线性整流函数ReLU可以在一定程度上解决训练进入梯度饱和区的问题。"""
    r"""$ \qquad \qquad \qquad $ 另一种思路是，我们可以让激活函数的输入分布保持在一个稳定状态来尽可能地避免它们陷入梯度饱和区，这也就是Normalization的思路。"""
    r"""$ \qquad $ 3.如何减缓Internal Covariate Shift?"""
    r"""$ \qquad \qquad $ ICS产生的原因是由于参数更新抵赖的网络中每一层输入值分布的改变，并且随着网络层数的加深而变得更加严重，因此我们可以通过固定每一层网络输入值的分布来减缓ICS问题。"""
    r"""$ \qquad \qquad $ (1)白化：白化(Whitening)是机器学习里面常用的一种规范化数据分布的方法，主要是PCA白化与ZCA白化。白化是对输入数据分布进行变换，进而达到以下两个目的，"""
    r"""$ \qquad \qquad \qquad $ 1)使得输入特征分布具有相同的均值与方差。其中PCA白化保证了所有特征分布均值为0，方差为1；而ZCA白化则保证了所有特征分布均值为0，方差相同。
"""
    r"""$ \qquad \qquad \qquad $ 2)去除特征之间的相关性。通过白化操作，我们可以减缓ICS的问题，进而固定了每一层网路输入分布，加速网络训练过程的收敛。"""
    r"""$ \qquad \qquad $ (2)Batch Normalization："""
    r"""$ \qquad \qquad \qquad $ 白化主要有以下两个问题：1)白化过程计算成本太高，并且在每一轮训练中的每一层我们都需要做如此高成本计算的白化操作；2)白化过程由于改变了网络每一层的分布，因而改变了网络层中本身数据的表达能力。底层网络学习到的参数信息会被白化操作丢失掉。"""
    r"""$ \qquad \qquad \qquad $ 思路：1)简化计算过程，单独对每个特征进行normalization，让每个特征都有均值为0，方差为1的分布。2)加线性变换操作，让这些数据再能够尽可能恢复本身的表达能力。"""
    r"""$ \qquad $ 4.算法步骤："""
    r"""$ \qquad \qquad $ 1.考虑一个batch的训练，传入m个训练样本，并关注网络中的某一层(第l层)$Z\in{R^{d_l\times{m}}}$。我们关注当前层的第j个维度，也就是第j个神经元节点，则有$Z_j\in{R^{d_l\times{m}}}$。我们当前维度进行规范化：
"""
    st.latex(r"""\mu_j=\frac{1}{m}\sum_{i=1}^m{Z_{j}^{(i)}}""")
    st.latex(r"""\sigma_j^2=\frac{1}{m}\sum_{i=1}^m{(Z_J^{(i)}-\mu_j)^2}""")
    st.latex(r"""\hat{Z}_j=\frac{Z_j-\mu_j}{\sqrt{\sigma_j^2+\varepsilon}}""")
    r"""$ \qquad \qquad $ 其中$\varepsilon$是为了防止方差为0产生无效计算。通过这种操作，用更加简化的方式来对数据进行规范。"""
    r"""$ \qquad \qquad $ 2.引入两个可学习(learnable)的参数$\gamma,\beta$。这两个参数的引入是为了恢复数据本身的表达能力，对规范化后的数据进行线性变换，即"""
    st.latex(r"""\tilde{Z}_j=\gamma_j{\hat{Z}_j}+\beta_j""")
    r"""$ \qquad \qquad $ 特别地，当$\gamma^2=\sigma^2,\beta=\mu$时，可以实现等价变换(identity transform)并且保留了原始输入特征的分布信息。"""
    r"""$ \qquad \qquad $ 同时，在进行normalization的过程中，由于我们的规范化操作会减去均值，因此偏置项b可以被忽略掉或可以被置为0，即"""
    st.latex(r"""BN(W\mu+b)=BN(W\mu)""")
    r"""$ \qquad $ 5.测试阶段如何使用Batch Normalization?"""
    r"""$ \qquad \qquad $ 在预测阶段，有可能只需要预测一个样本或很少的样本，没有像训练样本中那么多数据。此时$ \mu,\sigma^2 $的计算一定是有偏估计，这时应如何进行计算呢？"""
    r"""$ \qquad \qquad $ 1.利用BN训练好模型后，无保留每组mini_batch训练数据在网络中每一层的$ \mu_{𝑏𝑎𝑡𝑐ℎ} $，$ \sigma_{𝑏𝑎𝑡𝑐ℎ}^2 $。此时我们使用整个样本的统计量来对测试数据进行归一化，具体来说使用均值与方差的无偏估计："""
    st.latex(r"""\mu_{test}=E(\mu_{batch})""")
    st.latex(r"""\sigma_{test}^2=\frac{m}{m-1}E(\sigma_{batch}^2)""")
    r"""$ \qquad \qquad $ 得到每个特征的均值与方差的无偏估计后，我们对test数据采用同样的normalization方法："""
    st.latex(r"""BN(X_{test})=\gamma{\cdot{\frac{X_{test}-\mu_{test}}{\sqrt{\sigma_{test}^2+\varepsilon}}}}+\beta""")
    r"""$ \qquad \qquad $ 2.除了采用整体样本的无偏估计外，吴恩达指出可以对train阶段每个batch计算的mean/variance采用指数加权平均来得到test阶段mean/variance的估计。"""
    r"""$ \qquad $ 6.Batch Normalization的优势："""
    r"""$ \qquad \qquad $ (1)BN使得网络中每层输入数据的分布相对稳定，加速模型学习速度。"""
    r"""$ \qquad \qquad $ (2)BN使得模型对网络中的参数不那么敏感，简化调参过程，使得网络学习更加稳定。"""
    r"""$ \qquad \qquad $ (3)BN允许网络使用饱和性激活函数（例如sigmoid,tanh等），缓解梯度消失问题。"""
    r"""$ \qquad \qquad $ (4)BN具有一定的正则化效果。尽管每一个batch中的数据都是从总体样本中抽样得到，但不同mini-batch的均值与方差会有所不同，这就为网络的学习过程中增加了随机噪音，与Dropout通过关闭神经元网络训练带来噪音类似，在一定程度上对模型起到了正则化的效果。"""

    """- Dropout："""
    r"""$ \qquad $ Dropout的提出，源于2012年Hinton的一篇论文--《Improving neural networks by preventing co-adaptation of feature detectors》。论文中描述了当数据集较小时而神经网络模型较大较为复杂时，训练很容易产生过拟合，为了防止过拟合，可以通过组织特征检测器间的共同作用来提高模型性能。"""
    r"""$ \qquad $ Dropout通过在训练的过程中随机丢掉部分神经元来减小神经网络的规模从而防止过拟合。"""
    r"""$ \qquad $ Dropout工作原理：我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征，如图所示。 """
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片38.png', caption='图38')
    r"""$ \qquad $ 工作流程："""
    r"""$ \qquad \qquad $ 假设我们要训练这样一个神经网络，如图所示。"""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片39.png', caption='图39')
    r"""$ \qquad \qquad $ 输入是x，输出是y,正常的流程是：首先把x通过网络前向传播，然后把误差反向传播以决定如何更新参数让网络进行学习。使用Dropout之后，过程变成如下："""
    r"""$ \qquad \qquad $ （1）首先随机（临时）删掉网格中一半的隐藏神经元，输入输出神经元保持不变，图中虚线为部分临时被删除的神经元 。"""
    r"""$ \qquad \qquad $ （2）把输入x通过修改后的网格前向传播，然后把得到的损失结果通过修改的网络后向传播。一小批训练样本执行完这个过程后，在没有被删除的神经元上按照随机梯度下降法更新对应的参数(w,b)。"""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片40.png', caption='图40')
    r"""$ \qquad \qquad $ (3)然后继续重复这一过程："""
    r"""$ \qquad \qquad $ 恢复被删掉的神经元（此时被删除的神经元保持原样，而没有被删除的神经元已经有所更新）"""
    r"""$ \qquad \qquad $ 从隐藏层神经元中随机选择一个一半大小的子集临时删除掉（备份被删除神经元的参数）。"""
    r"""$ \qquad \qquad $对小批训练样本，先前向传播然后反向传播损失并根据梯度下降法更新参数(w,b)(没有被删除的那一部分参数得到更新，删除的神经元参数保持被删除前的结果)。"""
    r"""$ \qquad \qquad $ 不断重复过程(3)。"""
    r"""$ \qquad $ 为什么说Dropout可以解决过拟合问题？"""
    r"""$ \qquad \qquad $ 1. 取平均的作用。dropout掉不同的隐藏神经元就类似在训练不同的网络，随机删掉一半隐藏神经元导致网络结构已经不同，整个dropout过程就相当于对多个不同的神经网络取平均。而不同的网络产生不同的过拟合，一些互为“反向”的拟合相互低效就可以达到整体上减少过拟合。"""
    r"""$ \qquad \qquad $ 2. 减少神经元之间复杂的共适应关系。因为dropout程序导致两个神经元不一定每次都在一个dropout网络中出现。这样权值的更新不再依赖于有固定关系的隐含节点的共同作用，组织了某些特征仅仅在其它特征下才有效的情况。即使丢失特定的线索，它也应该可以从众多其它线索中学习一些共同的特征。"""
    r"""$ \qquad \qquad $ 3. 通过关闭神经元网络训练带来噪音，在一定程度上对模型起到了正则化的效果。"""
    r"""$ \qquad \qquad $"""
    """- Data Augmentation："""
    r"""$ \qquad $ 1.旋转 $\qquad$ 2.翻转 $\qquad$ 3.裁剪 $\qquad$ 4.缩放 $\qquad$ 5.平移 $\qquad$ 6.加噪声"""

    """- Early stopping："""
    r"""$ \qquad $ 提前停止（Early Stop）对于深度神经网络来说是一种简单有效的正则化方法。"""
    r"""$ \qquad $ 由于深度神经网络的拟合能力非常强，因此比较容易在训练集上过拟合。"""
    r"""$ \qquad $ 在使用梯度下降法进行优化时，我们可以使用一个和训练集独立的样本集合，称为验证集（Validation Set)，并用验证集上的错误来代替期望错误．当验证集上的错误率不再下降，就停止迭代。"""

with tab6:
    with st.expander('code'):
        st.code("""
import torch
import torch.nn as nn
import torchvision                          #torch中用来处理图像的库
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

#设置一些超参
num_epochs = 1        #训练的周期
batch_size = 100      #批训练的数量
learning_rate = 0.001 #学习率（0.1,0.01,0.001）

#导入训练数据
train_dataset = datasets.MNIST(root='E:/MNIST/',                #数据集保存路径
                               train=True,                      #是否作为训练集
                               transform=transforms.ToTensor(), #数据如何处理, 可以自己自定义
                               download=False)                  #路径下没有的话, 可以下载
                             
#导入测试数据
test_dataset = datasets.MNIST(root='E:/MNIST/',
                              train=False,
                              transform=transforms.ToTensor())                                                    
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, #分批
                                           batch_size=batch_size,
                                           shuffle=True)          #随机分批

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class MLP(nn.Module):                    #继承nn.module
	def __init__(self):
	super(MLP, self).__init__()      #继承的作用
        self.layer1 = nn.Linear(784,300)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(300,10)

        def forward(self,x):             #网络传播的结构
        x = x.reshape(-1, 28*28)
        x = self.layer1(x)
        x = self.relu(x)
        y = self.layer2(x)
        return y

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = mlp(images)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()                          #清零梯度
        loss.backward()                                #反向求梯度
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

#测试模型
mlp.eval()      #测试模式，关闭正则化
correct = 0
total = 0
for images, labels in test_loader:
    outputs = mlp(images)
    _, predicted = torch.max(outputs, 1)   #返回值和索引
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('测试准确率: {:.4f}'.format(100.0*correct/total))

    """)

    button = st.button('运行', key='MLP-mnist')
    if button:
        import torch
        import torch.nn as nn
        import torchvision  # torch中用来处理图像的库
        from torchvision import datasets, transforms
        import matplotlib.pyplot as plt
        # 设置一些超参
        num_epochs = 1  # 训练的周期
        batch_size = 100  # 批训练的数量
        learning_rate = 0.001  # 学习率（0.1,0.01,0.001）

        # 导入训练数据
        train_dataset = datasets.MNIST(root='E:/MNIST/',  # 数据集保存路径
                                       train=True,  # 是否作为训练集
                                       transform=transforms.ToTensor(),  # 数据如何处理, 可以自己自定义
                                       download=False)  # 路径下没有的话, 可以下载

        # 导入测试数据
        test_dataset = datasets.MNIST(root='E:/MNIST/',
                                      train=False,
                                      transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  # 分批
                                                   batch_size=batch_size,
                                                   shuffle=True)  # 随机分批

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)


        class MLP(nn.Module):  # 继承nn.module
            def __init__(self):
                super(MLP, self).__init__()  # 继承的作用

                self.layer1 = nn.Linear(784, 300)
                self.relu = nn.ReLU()
                self.layer2 = nn.Linear(300, 10)

                def forward(self, x):  # 网络传播的结构
                    x = x.reshape(-1, 28 * 28)

                x = self.layer1(x)
                x = self.relu(x)
                y = self.layer2(x)
                return y


        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                outputs = mlp(images)
                loss = loss_func(outputs, labels)
                optimizer.zero_grad()  # 清零梯度
                loss.backward()  # 反向求梯度
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

        # 测试模型
        mlp.eval()  # 测试模式，关闭正则化
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = mlp(images)
            _, predicted = torch.max(outputs, 1)  # 返回值和索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('测试准确率: {:.4f}'.format(100.0 * correct / total))


