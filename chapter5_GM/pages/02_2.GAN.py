import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tab1, tab2 = st.tabs(['1.GAN', '2.GAN的变种'])

with tab1:

    """1. GAN是一种生成模型，它通过对抗过程估计生成模型，它也是一种无监督学习模型。"""
    """2. GAN和VAE的不同之处在于它是implicit density estimate，不需要做显式分布假设。"""
    """3. GAN由两部分组成，生成器G和判别器D，生成器G生成虚假样本，判别器D对虚假样本和真实样本进行判别，二者交替学习，最终得到越来越逼真的生成模型。"""
    """$ \qquad $ GAN的损失函数如下："""
    st.latex("""
    min_Gmax_DV(D,G) = E_{x\sim{p_{data}(x)}}[logD(x)]+E_{z\sim{p_z(z)}}[log(1-D(G(z)))]
    """)
    """$ \qquad \qquad $ 其中，其中，G代表生成器， D代表判别器， x代表真实数据，$p_{data}$代表真实数据概率密度分布，z代表了随机输入数据，该数据是随机高斯噪声。"""
    """$ \qquad $ 从上式可以看出，从判别器D角度来看，判别器D希望能尽可能区分真实样本x和虚假样本G(z)，因此D(x)必须尽可能大，D(G(z))尽可能小，也就是V(D,G)整体尽可能大。从生成器的角度来看，生成器G希望自己生成的虚假数据G(z)可以尽可能骗过判别器D,也就是希望D(G(z))尽可能大，也就是V(D,G)整体尽可能小。GAN的两个模块在训练相互对抗，最后达到全局最优。"""

    """- GAN-网络结构:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片10.jpg', caption='图10')
    """- GAN的优缺点:"""
    """$ \qquad $ 1.优点："""
    """$ \qquad \qquad $ GAN是一种生成式模型，相比较其他生成模型（玻尔兹曼机和GSNs）只用到了反向传播,而不需要复杂的马尔科夫链。"""
    """$ \qquad \qquad $ 相比其他所有模型, GAN可以产生更加清晰，真实的样本。"""
    """$ \qquad \qquad $ GAN采用的是一种无监督的学习方式训练，可以被广泛用在无监督学习和半监督学习领域。"""
    """$ \qquad \qquad $ 相比于变分自编码器, GANs没有引入任何决定性偏置( deterministic bias),变分方法引入决定性偏置,因为他们优化对数似然的下界,而不是似然度本身,这看起来导致了VAEs生成的实例比GANs更模糊。"""
    """$ \qquad \qquad $ 相比VAE, GANs没有变分下界,如果鉴别器训练良好,那么生成器可以完美的学习到训练样本的分布。换句话说,GANs是渐进一致的,但是VAE是有偏差的。"""
    """$ \qquad $ 2.缺点："""
    """$ \qquad \qquad $ 训练过程不稳定，很难收敛-博弈"""
    """$ \qquad \qquad $ 不适合处理文本数据；"""
    """$ \qquad \qquad $ 梯度消失问题-wgan"""
    """$ \qquad \qquad $ 模式崩溃问题-AE-OT"""

with tab2:
    """ 1.CGAN, DCGAN, """
    """ 2.Info-GAN, Style-GAN，Cycle-GAN..."""
    """ 3.WGAN、WGAN-GP"""
    """ 4.AE-OT"""
    """- Conditional GAN:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片11.png', caption='图11')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片12.jpg', caption='图12')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片13.png', caption='图13')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片14.png', caption='图14')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片15.jpg', caption='图15')
    """- Deep Convolutional GAN:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片16.png', caption='图16')
    """$ \qquad $ DCGAN 的判别器和生成器都使用了卷积神经网络（CNN）来替代GAN 中的多层感知机，同时为了使整个网络可微，拿掉了CNN 中的池化层，另外将全连接层以全局池化层替代以减轻计算量。"""
    """- Info-GAN:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片17.png', caption='图17')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片18.png', caption='图18')


    """- Style-GAN:"""
    """$ \qquad $ 受风格迁移启发，styleGAN重新设计了生成器网络结构，并试图来控制图像生成过程：生成器从学习到的常量输入开始，基于潜码调整每个卷积层的图像“风格”，从而直接控制图像特征；另外，结合直接注入网络的噪声，可以更改所生成图像中的随机属性（例如雀斑、头发）。styleGAN可以一定程度上实现无监督式地属性分离。进行一些风格混合或插值地操作。"""
    """$ \qquad $ 基于风格驱动的生成器："""
    """$ \qquad \qquad $ 在以往，潜码仅喂入生成器的输入层。而StyleGAN是设计了一个非线性映射网络f对输入的潜在空间Z中的潜码z进行加工转换：$Z\rightarrow{W}(w\in{W})$。为简单起见，将两个空间的维数都设置为512，并且使用8层MLP实现映射f。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片97.jpg', caption='图97')
    """$ \qquad \qquad $ 非线性映射也就是学习到的仿射变换将w定制化为风格$y=(y_s,y_b)$，它们在接下来的生成网络g中每个卷积层之后被用于控制自适应示例归一化(AdalN)。AdalN运算定义为如下式所示。
    其中，每个特征图$x_i$分别归一化，然后使用y中对应的标量。因此，y的维数是该层特征图的两倍。与风格迁移不同，StyleGAN是从向量w而不是风格图像计算而来。"""
    st.latex(r"""AdaIN(x_i,y)=y_{s,i}\frac{x_i-\mu(x_i)}{\sigma{(x_i)}}+y_{b,i}""")
    """$ \qquad \qquad $ 最后，通过对生成器引入噪声输入，提供了一种生成随机（多样性）细节的方法。噪声输入是由不相关的高斯噪声组成的单通道数据，它们被馈赠送到生成网络的每一层。"""
    """- Cycle-GAN:"""
    """$ \qquad $ CycleGan是一个神经网络，可以学习两个域之间的两个数据转换函数。 其中之一是G(x)。 它将给定样本$x\in{X}$转换为域Y的元素。第二个是F(y)，它将样本元素$y\in{Y}$转换为域X的元素。"""
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片19.png', caption='图19')
    with col2:
        st.image('./chapter5_GM/pages/图片/图片20.png', caption='图20')
    """- 损失函数:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片21.png', caption='图21')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片22.png', caption='图22')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片23.png', caption='图23')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片24.png', caption='图24')
    """- GAN模型分析:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片25.png', caption='图25')
    """$ \qquad $ 假设$𝑝_𝑟(𝑥)，𝑝_{\thta}(𝑥)$已知，则最优的判别器为:"""
    _, col1, _ = st.columns([1, 1, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片26.png', caption='图26')
    """$ \qquad $ 目标函数变为："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片27.png', caption='图27')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片28.png', caption='图28')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片29.png', caption='图29')
    """- 不稳定性：生成网络的梯度消失"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片30.png', caption='图30')
    _, col1, _ = st.columns([1, 1, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片31.png', caption='图31')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片32.png', caption='图32')
    """$ \qquad $ 在生成对抗网络中，当判断网络为最优时，生成网络的优化目标是最小化真实分布$𝑝_𝑟(𝑥)$和模型分布$𝑝_{\theta}(𝑥)$之间的JS散度。当两个分布相同时，JS散度为0，最优生成网络对应的损失为−2log2。"""
    """$ \qquad $ 使用JS散度来训练生成对抗网络的一个问题是当两个分布没有重叠时，它们之间的JS散度恒等于常数log2。对生成网络来说，目标函数关于参数的梯度为0。"""
    """- WGAN:"""
    """$ \qquad $ Wasserstein距离用于衡量两个分布之间的距离。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片33.png', caption='图33')
    """$ \qquad $ 其中$\gamma{(q_1,q_2)}$是边际分布为$𝑞_1,𝑞_2$的所有可能的联合分布集合，d(x,y)为x和y的距离，比如$ℓ_𝑝$距离等。"""
    """$ \qquad $ Wasserstein距离相比KL散度和JS散度的优势在于：即使两个分布没有重叠或者重叠非常少，Wasserstein距离仍然能反映两个分布的远近。"""
    """- Kantorovich-Rubinstein 对偶定理:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片34.png', caption='图34')

    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片35.png', caption='图35')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片36.png', caption='图36')
    """- 模型坍塌：生成网络的“错误”目标"""
    """$ \qquad $ 生成网络的目标函数。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片37.png', caption='图37')
    """$ \qquad $ 其中后两项和生成网络无关，因此"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片38.png', caption='图38')
    """- 前向散度和逆向KL散度:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片39.png', caption='图39')
    _,col1, col2 = st.columns([1, 1, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片40.png', caption='图40')
    with col1:
        st.image('./chapter5_GM/pages/图片/图片41.png', caption='图41')
    """$ \qquad $ 前向KL散度会鼓励模型分布$p_{\theta}(x)$尽可能覆盖所有真实分布$p_r(x)>0$的点，而不用回避$p_r(x)≈0$的点。逆向KL散度会鼓励模型分布$p_{\theta}(x)$尽可能避开所有真实分布$p_r(x)≈0$的点，而不需要考虑是否覆盖所有布$p_r(x)>0$的点。"""

    """- WGAN-GP:"""
    """$ \qquad $ WGAN存在的两个问题："""
    """$ \qquad \qquad $ 1.WGAN在处理Lipschitz限制条件时直接采用了weight clipping，即每当更新完一次判别器的参数之后，就检查判别器的所有参数的绝对值有没有超过一个阈值，比如0.01。有的话就把这些参数clip回[-0.01,0.01]范围内。通过在训练过程中保证判别器的所有参数有界，就保证了判别器不能对两个有略微不同的样本在判别器上不会差异差异过大，从而间接实现了Lipschitz限制。
    实际训练上判别器loss希望尽可能拉大真假样本的分数差，然而weight clipping独立地限制每一个网络参数的取值范围，在这种情况下最优的策略就是尽可能让所有参数走极端，要么取最大值（如0.01），要么取最小值（如-0.01），文章通过实验验证了猜测如下图所示判别器的参数几乎都集中在最大值和最小值上。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片98.png', caption='图98')

    """$ \qquad \qquad $ 2.weight clipping会导致很容易一不小心就梯度消失或梯度爆炸。原因是判别器是一个多层网络，如果把clipping threshold设的稍微笑了一点，每经过一层网络，梯度就变小一点点，多层之后就会指数衰减；反之，如果设的稍微大了一点，每经过一层网络，梯度就会变大一点点，多层之后就会指数爆炸。只有设的不大不小，才能让生成器获得恰到好处的回传梯度。然而在实际应用中这个平衡区域可能很狭窄，就会给调参工作带来麻烦。文章也通过实验展示了这个问题，下图中横轴代表判别器从低到高第几层，纵轴代表梯度回传到这一层之后的尺度大小（注意纵轴是对数刻度）。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片99.png', caption='图99')

    """$ \qquad $ 解决方案：添加梯度惩罚。Lipschitz限制是要求判别器的梯度不超过K,梯度惩罚设置一个额外的loss项来实现梯度与K之间的联系，这就是梯度盛饭的核心所在，下图为引入梯度盛饭后WGAN-GP的算法图。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片100.png', caption='图100')

    """$ \qquad $ 梯度惩罚的选取并不是在全网络下，仅仅是在真假分布之间抽样处理，下图为处理过程。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片101.png', caption='图101')
    """$ \qquad $ WGAN-GP的最终目标函数： """
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片102.png', caption='图102')
    """$ \qquad $ WGAN-GP的创新点也就在目标函数的第二项上，由于模型是对每个独立样本独立地施加梯度惩罚，所以判别器的模型架构中不能使用Batch Normalization，因为它会引入同个batch中不同样本的相互依赖关系。"""

    """- AE-OT:"""
    """$ \qquad $ 目前的深度神经网络只能够逼近连续映射，而传输映射是具有间断点的非连续映射，换言之，GAN训练过程中，目标映射不在DNN的可表示泛函空间之中，这一显而易见的矛盾导致了收敛困难；"""
    """$ \qquad $ 如果目标概率测度的支集具有多个联通分支，GAN训练得到的又是连续映射，则有可能连续映射的值域集中在某一个连通分支上，这就是模式崩溃（mode collapse）；"""
    """$ \qquad $ 如果强行用一个连续映射来覆盖所有的连通分支，那么这一连续映射的值域必然会覆盖图片之外的一些区域，即GAN会生成一些没有现实意义的图片。这给出了GAN模式崩溃的直接解释。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片42.png', caption='图42')
    """$ \qquad $ 基于真实数据的流形分布假设，我们将深度学习的主要任务分解为学习流形结构和概率变换两部分；概率变换可以用最优传输理论来解释和实现。基于Brenier理论，我们发现GAN模型中的生成器D和判别器G计算的函数彼此可以相互表示，因此生成器和判别器应该交流中间计算结果，用合作代替竞争。Brenier理论等价于蒙日-安培方程，蒙日-安培方程正则性理论表明：如果目标概率分布的支集非凸，那么存在零测度的奇异点集，传输映射在奇异点处间断。而传统深度神经网络只能逼近连续映射，这一矛盾造成了模式崩溃。"""
    """$ \qquad $ 通过计算Brenier势能函数，并且判定奇异点集，我们可以避免模式崩溃。"""
    """- GAN的应用:"""
    """$ \qquad $ 各种各样的计算机视觉任务中如："""
    """$ \qquad \qquad $ （1）人脸图像生成"""
    """$ \qquad \qquad $ （2）风格迁移"""
    """$ \qquad \qquad $ （3）图像去噪、超分辨"""
    """$ \qquad \qquad $ （4）图像补全"""
    """$ \qquad \qquad $ （5）...."""

