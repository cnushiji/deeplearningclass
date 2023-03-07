import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tab1, tab2, tab3 = st.tabs(['1.数学原理', '2.模型构建', '3.简化损失函数'])

with tab1:
    """- 1.生成模型：给定从一个分布中观察到的样本x,生成模型的目标是学习建模真实的数据分布p(x)。学习到之后，我们可以任意地从近似模型中生成新的样本。"""
    """扩散模型(diffusion model)属于无监督生成模型。下面我们给出研究至今的生成模型的分类："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片118.jpg', caption='图118 生成模型分类')
    """ 它们的不同在于建模方式，GAN对复杂分布的抽样过程建模并以对抗性的方式进行学习；自回归模型、正则化流和变分自动编码器（VAEs）基于”似然性“，对观察到的数据样本学习到一个分配高似然性的模型。"""
    """ 第3种是基于能量的建模，一个分布被学习为一个任意灵活的能量函数，再被正则化。基于分数的生成模型与之高度相关，具体地，不是学习建模能量函数本身，而是学习一个基于能量的模型的分数作为神经网络。"""

    """ $ \\textbf{2.变分扩散模型(VDM)} $"""
    """ 1.马尔可夫性质(Markov property)是概率论中的一个概念，因俄国数学家安德雷·马尔可夫得名。当一个随机过程在给定现在状态及所有过去状态情况下，其未来状态的条件概率分布仅依赖于当前状态。"""
    """ 2.我们可以将VDM理解为有三个关键限制的马尔可夫层次的变分自编码器。"""
    """ $ \qquad $ 1.隐藏层的维度完全等于数据维度。"""
    """ $ \qquad $ 2.不去学习潜在编码器在每个时间步长的结构，它被预先定义为一个线性高斯模型。另一种说法，它是一个以前一个时间步长的输出为中心的高斯分布。"""
    """ $ \qquad $ 3.潜在编码器的高斯参数随时间变化，使得潜在编码的最终时间步长T的分布是标准高斯分布。"""

    """ 3.第1个限制，可以表示为如下公式："""
    st.latex(r"""
    q(x_{1:T}|x_0)=\prod \limits_{t=1}^T{q(x_t|x_{t-1})}
    """)
    """ 4.第2个限制，编码器的每个隐藏变量的分布都是一个围绕其先前层次的隐藏变量为中心的高斯分布。即与马尔可夫HVAE不同，我们不学习每个时间步长t的编码器的结构，而是被固定为一个线性高斯模型，其均值和标准差被预先设定为超参数或学习参数。
    均值：$\mu_t(x_t)=\sqrt{\\alpha_t}x_{t-1}$,方差$\sum_t(x_t)=(a-\\alpha_t)I$,其中对于系数的选择，使隐藏变量的方差保持在相似的尺度上，即编码过程保持方差。
    $\\alpha_t$是一个（潜在的可学习的）系数，为了灵活性，它可以随着层次深度t而变化，这样可设定最终的隐藏$p(x_T)$是一个标准的高斯分布。综上，编码器的转换在数学上可公式化为："""
    st.latex(r"""
    q(x_t|x_{t-1})=N(x_t;\sqrt{\alpha_t}(x_{t-1}),(1-\alpha_t)I)
    """)
    """ 5.第3个限制，我们可以将VDM的联合分布写为："""
    st.latex(r"""
    p(x_{0:T})=p(x_T)\prod \limits_{t=1}^T{p_{\theta}(x_{t-1}|x_t)},where p(x_T)=N(x_T;0,I) 
    """)

    """- $\\textbf{总之，以上3个限制描述的是图像输入随时间推移的稳定噪声，我们通过添加高斯噪声（也可以是其它噪声）逐步破坏图像，知道它变成一个标准高斯噪声。}$从视觉上看，如图所示："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片119.png', caption='图119')
    """ 如图所示，每个$q(x_t|x_{t-1})$被建模为一个高斯分布，它使用前一个状态的输出作为其平均值。"""

    """ 注意，$q(x_t|x_{t-1})$不再被$\phi$参数化，因为它们被完全建模为高斯分布，在每个时间步长都定义好了均值和方差。因此，只对学习条件$p_{\theta}(x_{t-1}|x_t)$感兴趣，可以模拟新的数据。
    在优化VDM后，从$p(x_T)$中采样高斯噪声，迭代运行去噪转换$p_{\\theta}(x_{t-1}|x_T)$t步生成一个新的$x_0$。"""

    """ 6.VDM的优化1——等价于最大化ELBO"""
    """ 在数学上，利用联合分布恢复数据p(x)又两种方法，1："""
    st.latex(r"""
    p(x)=\int{p(x,z)dz}
    """)
    """$ \qquad $ 其中，z是隐藏变量。"""

    """ 2.概率的链式法则："""
    st.latex(r"""
    p(x)=\frac{p(x,z)}{p(z|x)}
    """)
    """ VDM的优化推导："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片120.png', caption='图120')
    """$ \qquad $ 1.$E_{q(x_1|x_0)}[logp_{\\theta}(x_0|x_1)]$是重建项，给定第1步潜码$x_1$，预测原始数据样本的对数概率。"""
    """$ \qquad $ 2.$E_{q(x_{T-1})[D_{KL}(q(X_T)|x_{T-1})||p(x_T)]}$是先验匹配项。它是利用$q(x_{T-1}|x_0)=q(x_{T-1}|x_{T-2})$和KL散度公式计算得出。当最终的潜在分布与高斯先验相匹配时，它被最小化。这项不需要优化，因它没有可训练的参数。如果T足够大，这项会变成0."""
    """$ \qquad $ 3.$E_{q(x_{t-1},x_{t+1}|x_0)}[D_{KL}(q(x_t|x_{t-1})||p_{\\theta}(x_t|x_{t+1}))]$是一致性项。跟2一样，利用$q(x_{T-1}|x_0)=q(x_{T-1}|x_{T-2})$和KL散度公式计算得出。它促进正向过程和反向过程的$x_t$处的分布一致。即对于每个中间时间步，去噪步中的噪声图像应与相应时间步的从干净图像得到的去噪图像一致，数学上用KL散度促进一致性。当我们训练这两项一致时，该项被最小化。"""

    """$ \qquad $ 从视觉上看，这种解释如图所示："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片121.png', caption='图121')
    """$ \qquad $ 优化VDM的成本主要在第3项上，因为我们必须优化所有的时间步长。"""

    """ $\\textbf{我们刚刚推导出的VDM的优化可能是次优的，因为一致性项基第3项是对每个时间步的两个随机变量求期望，它的蒙特卡洛估计的方差可能高于每个时间步只使用一个随机变量的项。又它是将T-1个一致性项相加得到，这对于充分大的T,VDM的最终估计可能有较大的方差。}$"""

    """ 6.VDM的优化2——每个项都被计算为一次只对一个随机变量的期望"""
    """- 关键点：将编码器转换重写为$q(x_t|x_{t-1})=q(x_t|x_{t-1},x_0)$,其中由于马尔可夫性质，额外的条件项是多余的。根据贝叶斯规则，我们可以将转换改写为："""
    st.latex(r"""
    q(x_t|x_{t-1},x_0)=\frac{q(x_{t-1}|x_t,x_0)q(x_t|x_0)}{q(x_{t-1}|x_0)}
    """)
    """ 如此，我们可以重写优化1中的推到："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片122.png', caption='图122')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片123.png', caption='图123')
    """$ \qquad $ 1.$E_{q(x_1|x_0)}[logp_{\\theta}(x_0|x_1)]$是重建项，这个项可以用蒙特卡洛估计进行估计和近似。"""
    """$ \qquad $ 2.$D_{KL}(q(x_T|x_0)||p(x_T))$表示最终的噪声输入分布与标准高斯先验的接近程度。它也没有可训练的参数，故可以假设为0。"""
    """$ \qquad $ 3.$E_{q(x_t|x_0)}[D_{KL}(q(x_{t-1}|x_t,x_0))]$是去噪匹配项，$q(x_{t-1}|x_t,x_0)$是地面真值转换步，这个过度步骤可以作为地面真值信号，因为它定义了应该如何去噪噪声图像$x_t$才能到达最终的完全去噪的$x_0$。因此当两个去噪步匹配时，这项的KL散度是最小化的。"""
    """$\\textbf{以上2个VDM的优化推导只是用了马尔可夫假设。}$"""
    """$ \qquad $ 在任意复杂的马尔可夫HVAEs中，由于同时学习编码器的复杂性，每个KL散度项$E_{q(x_t|x_0)}[D_{KL}(q(x_{t-1}|x_t,x_0))]$都很难被最小化。在VDM中，我们可以用高斯转移假设，使优化易于处理。由贝叶斯规则，我们有："""
    st.latex(r"""
    q(x_{t-1}|x_t,x_0)=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}
    """)
    """$ \qquad $ 根据我们有关编码转移的假设，有$q(x_t|x_{t-1},x_0)=q(x_t|x_{t-1})=N(x_t;\sqrt{\\alpha_t},(1-\\alpha_tI))$。接下来使对$q(x_t|x_0)$和$q(x_{t-1}|x_0)$的形式的推导，利用VDM的编码器转换是线性高斯模型，在重参数化技巧下，样本$x_t\sim{q(x_t|x_{t-1})}$可以被重写为："""
    st.latex(r"""
    x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon \qquad with \epsilon \sim{N(\epsilon;0,I)}
    """)
    """$ \qquad $ 类似地，样本$x_{t-1}\sim{q(x_{t-1}|x_{t-2})}$可以被重写为："""
    st.latex(r"""
    x_{t-1}=\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}\epsilon} \qquad \epsilon \sim{N(\epsilon;0,I)}
    """)
    """$ \qquad $ 接下来我们对$q(x_t|x_0)$应用重参数化技巧。假设我们可以访问2T个随机噪声变量$\left\{\epsilon^{*}_t,\epsilon_t \\right\}_{t=0}^T \overset{iid} \sim{N(\epsilon;0,I)}$,于是对任意样本$x_t\sim{q(x_t|x_0)}$，我们可以将它重写为："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片124.png', caption='图124')
    """$ \qquad $ 其中，我们利用了两个独立的高斯随机变量的和仍然是一个高斯分布的事实，其中均值是两个均值的和，方差是这两个方差的和。将$\sqrt{1-\\alpha_t}\epsilon_{t-1}^{*}$解释为高斯分布$N(0,(1-\\alpha_t)I)$的一个样本，将$\sqrt{\\alpha_t-\\alpha_t{\\alpha_{t-1}}}\epsilon_{t-2}^{*}$看作高斯分布$N(0,(1-\\alpha_t-\\alpha_t{\\alpha_{t-1}})I)$的样本，然后我们可以将它们的和
    看作一个从高斯分布$N(0,(1-\\alpha_t+\\alpha_t-\\alpha_t{\\alpha_{t-1}})I)=N(0,(1-\\alpha_t{\\alpha_{t-1}})I)$中采样的随机变量。然后这个分布的样本可以使用重参数化技巧表示为：$\sqrt{1-\\alpha_t{\\alpha_{t-1}}}\epsilon_{t-2}$。"""
    """- 至此，我们推导出了$q(x_t|x_0)$的高斯形式。同理，我们也可以得到$q(x_{t-1}|x_0)}$的高斯重参数化形式。"""
    """ 现在贝叶斯公式所需的3个形式都得到，接下来替换贝叶斯形式如下："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片125.png', caption='图125')
    """$ \qquad $ 其中，$C(x_t,x_0)$是关于$x_{t-1}$的常数项。"""
    """- 至此已经证明了每一个时间步，$x_{t-1}\sim{q(x_{t-1}|x_t,x_0)}$是正态分布的，均值$\mu_q(x_t,x_0)$是$x_t$和$x_0$的函数，方差$\sum_q(t)$是$\\alpha$系数的函数。这些$\\alpha$系数在每个时间步是固定的和已知的。当它们被建模为超参数时，要么是永久设置的，要么被认为是一个网络建模它们时的当前输出。"""
    """$ \qquad $ 根据$\sum_q(t)$，可以将方差重写为$\sum_q(t)=\sigma_q^2(t)I$，其中"""
    st.latex(r"""
    \sigma_q^2(t)=\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}
    """)
    """$ \qquad $ 为了将近似去噪转移步$p_{\\theta}(x_{t-1}|x_t)$尽可能地与地面真值去噪转移步$q(x_{t-1}|x_t,x_0)$尽可能地匹配，我们还可以将近似去噪转移步建模为高斯分布。"""

    """$ \qquad $ 1.由于所有$\\alpha$项在每个时间步都被冻结，我们可以将近似去噪转换步的方法也构造为$\sum_q(t)=\sigma_q^2(t)$。"""
    """$ \qquad $ 2.因为$p_{\\theta}(x_{t-1}|x_t)$对$x_0$没有条件，我们需要将均值$\mu_{\\theta}(x_t,t)$参数化为$x_t$的函数。"""
    """$ \qquad $ 两个高斯分布之间的KL散度为："""
    st.latex(r"""
    D_{KL}(N(x;\mu_x,\sum_x)||N(y;\mu_y,\sum_y))=\frac{1}{2}[log\frac{|\sum_y|}{|\sum_x|}-d + tr(\sum_y^{-1}\sum_x)+(\mu_y-\mu_x)^T\sum_t^{-1}(\mu_y-\mu_x)]
    """)
    """$ \qquad $ 我们可以将2个高斯分布的方法精确匹配，将优化KL散度简化为最小化两个分布的平均值之间的差异。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片126.png', caption='图126')
    """$ \qquad $ 我们想要优化$\mu_{\\theta}(x_t,t)$匹配$\mu_q(x_t,x_0)$。从图125中可得二者的形式："""
    st.latex(r"""
    \mu_q(x_t,x_0)=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t+\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)x_0}{1-\bar{\alpha}_t}
    """)
    """$ \qquad $ 由于$\mu_{\\theta}(x_t,t)$也是以$x_t$为条件，我们可以通过将$\mu_q(x_t,x_0)$为以下形式来紧密配合它："""
    st.latex(r"""
    \mu_{\theta}(x_t,t)=\frac{\sqrt{\alpha}_t(1-\sqrt{\alpha}_{t-1})x_t+\sqrt{\bar{\alpha}}(1-\alpha_t)\hat{x}_{\theta}{x_t,t}}{1-\bar{\alpha}_t}
    """)
    """$ \qquad $ 其中，$\hat{x}_{\\theta}(x_t,t)$是由一个神经网络参数化，该神经网络试图从有噪声的图像$x_T$和时间步t中预测$x_0$。然后，将优化问题简化为："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片127.png', caption='图127')
    """$\\textbf{因此，优化VDM相当于学习一个能从任意噪声版本预测原始地面真值图像的神经网络。}$"""
    """- 同时VDM优化目标2可以被它的去噪匹配项近似。于是VDM优化目标进一步表示为："""
    st.latex(r"""
    \mathop{argmin}\limits_{\theta}E_{t\sim{U \left\{2,T \right\}}}[E_{q(x_t|x_0)}[D_{KL}(q(x_{t-1}|x_t,x_0)||p_{\theta}(x_{t-1}|x_t))]]
    """)
    """$ \qquad $ 然后可以在时间步长使用随机采样进行优化。"""

    """$\\textbf{3.学习扩散噪声参数}$"""
    """ 方法1.一种可能的方法是使用一个参数为$\eta$的神经网络$\hat{\\alpha}_{\eta}(t)$来建模$\\alpha_t$。但是效率低，因为在每个时间步t上必须推理多次才能计算$\hat{\\alpha}_t$。减轻这种计算成本的方式是缓存。"""
    """ 方法2.将方差方程$\sigma_q^2(t)$代入每个时间不长的优化目标："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片128.png', caption='图128')
    """$ \qquad $ 回想，$q(x_t|x_0)=N(x_t;\sqrt{bar{\\alpha}_t}x_0,(1-\\bar{\\alpha}_t)I)$。然后根据信噪比的定义$SNR=\\frac{\mu^2}{\sigma^2}$，我们可以将在每个时间步t的SNR写为："""
    st.latex(r"""
    SNR(t)=\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}
    """)
    """ 然后我们可以将图128的结果简化为："""
    st.latex(r"""
    \frac{1}{2\sigma_q^2(t)}\frac{\sqrt{\alpha}_{t-1}(1-\alpha_t)^2}{(1-\bar{\alpha}_t)^2}[\Vert \hat{x}_{\theta}(x_t,t)-x_0 \Vert_2^2]=\frac{1}{2}(SNR(t-1)-SNR(t))[\Vert \hat{x}_{\theta}(x_t,t)-x_0 \Vert_2^2]
    """)
    """$ \qquad $ 信噪比表示原始信号与所存在的噪声量之间的比值，较高的信噪比代表越多的信号，而较低的信噪比代表越多的噪声。在扩散模型中，我们要求信噪比随着时间步长t的增加而单调降低，这形式化了扰动输入$x_t$随着时间变得越来越嘈杂，直到它在$t+T$时与标准高斯分布相同。"""
    """$ \qquad $ 由于上述目标的简化，我们可以使用神经网络直接参数化每个时间步长的信噪比，并将其与扩散模型共同学习。由于信噪比必须随着时间单调下降，可以将其表示为："""
    st.latex(r"""
    SNR(t)=exp(-\omega_{\eta}(t))
    """)
    """$ \qquad $ 其中，$\omega_{\eta}(t)$被建模为一个参数为$\eta$的单调递增的神经网络。负的$\omega_{\eta}(t)$是一个单调递减的函数，指数满足了得到的项为正。因此可以得到："""
    st.latex(r"""
    \frac{\bar{\alpha}_t}{1-\bar{\alpha}}=exp(-\omega_{\eta}(t)),所以\bar{\alpha}_t=sigmoid(-\omega_{\eta}(t)),所以1-\bar{\alpha}_t=sigmoid(\omega_{\eta}(t))
    """)
    """$ \qquad $ 这些是有必要的，例如在优化过程中，可以使用重新参数化技巧从输入$x_0$创建任意噪声$x_t$。"""

    """$\\textbf{三种等价解释} $"""
    """ 1.上述已经证明，变分扩散模型可以通过简单地学习神经网络来从任意噪声版本$x_t$及其时间指数t预测原始再燃图像$x_0$。"""
    """ 2.首先利用重参数化技巧，将之前对$q(x_t|x_0)$的推导形式重新排列方程表示为："""
    st.latex(r"""
    x_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_0}{\sqrt{\bar{\alpha}_t}}
    """)
    """$ \qquad $ 将其代入我们之前推导的地面真值去噪转移均值$\mu_q(x_t,x_0)$中，可以得到："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片129.png', caption='图129')
    """ 因此，我们可以将近似去噪转移均值$\mu_{\\theta}(x_t,t)$设为："""
    st.latex(r"""
    \mu_{\theta}(x_t,t)=\frac{1}{\sqrt{\alpha}_t}x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t}}\hat{\epsilon}_{\theta}(x_t,t)
    """)
    """$ \qquad $ 然后，相应的优化问题变为："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片130.png', caption='图130')
    """$ \qquad $ 这里，$\hat{\epsilon}(x_t,t)$是一个神经网络，它学习预测能从$x_0$决定$x_t$的源噪声$\epsilon_0\sim{N(\epsilon;0,I)}$。"""
    """- 因此，我们证明了通过预测原始图像$x_0$学习VDM相当于学习预测噪声。在实际工作中也发现，预测噪声会导致更好的性能。"""

    """ 3.采用Tweedie的公式，一个指数簇分布的真值平均值，给定从它中抽取的样本，可以通过样本的最大似然估计（即经验平均值）加上一些涉及估计的分数的修正项来估计。在只有一个观察样本的情况下，经验平均值就是样本本身，通常用于减轻样本偏差；如果观察到的样本都位于潜在分布的一端，则福份属变得很大，并将样本的naive最大似然估计修正为真实均值。"""
    """$ \qquad $ 数学上，对于一个高斯变量$z\sim{N(z;\mu_z,\sum_z)}$，Tweedie的公式表明，"""
    st.latex(r"""
    E[\mu_z|z]=z+\sum_z\nabla{logp(z)}
    """)
    """$ \qquad $ 在这种情况下，我们用它来预测给定其样本$x_t$的真实后验均值。我们已知，"""
    st.latex(r"""
    q(x_t|x_0)=N(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
    """)
    """$ \qquad $ 由Tweedie的公式，"""
    st.latex(r"""
    E[\mu_{x_t}|x_t]=x_t+(1-\bar{\alpha}_t)\nabla_{x_t}logp(x_t)
    """)
    """$ \qquad $ 然后，我们将$\\nabla_{x_t}logp(x_t)$简写成$\\nabla{logp(x_t)}$。根据Tweedie的公式，由$\mu_{x_t}=\sqrt{\\bar{\\alpha}_t}x_0$生成的$x_t$的真值平均值的最佳估计值定义为："""
    st.latex(r"""
    \sqrt{\bar{\alpha}_t}x_0=x_t+(1-\bar{\alpha}_t)\nabla{logp(x_t)},所以x_0=\frac{x_t+(1-\bar{\alpha}_t)\nabla{logp(x_t)}}{\sqrt{\bar{\alpha}_t}}
    """)
    """$ \qquad $ 然后，将新得到的$x_0$代入我们的地面真值去噪变换均值$\mu_q(x_t,x_0)$，进行推导："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片131.png', caption='图131')

    """$ \qquad $ 因此，我们可以设定我们的近似去噪转移均值$\mu_{\\theta}(x_t,t)$为："""
    st.latex(r"""
    \mu_{\theta}(x_t,t)=\frac{1}{\sqrt{\alpha_t}}x_t+\frac{1-\alpha_t}{\sqrt{\alpha_t}}s_{\theta(x_t,t)}
    """)
    """$ \qquad $ 然后，相应的优化问题变为："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片132.png', caption='图132')
    """$ \qquad $ 这里，$s_{\\theta}(x_t,t)$是一个神经网络，它学习预测分数函数$\\nabla_{x_t}logp(x_t)$，它是数据空间中的$x_t$对于任何任意噪声水平t的梯度。"""
    """$ \qquad $ 到此为止，有人应该看出分数函数$\\nabla{logp(x_t)}$看起来与源噪声$\epsilon$在形式上非常相似。这可以结合Tweedie的公式和重参数化技巧显示："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片133.png', caption='图133')
    """$ \qquad $ 通过证明，这两项被一个随时间变化的常数因子抵消了。分数函数测量如何在数据空间中移动以最大化对数概率。即，由于源噪声被添加到自然图像中以破坏它，向相反的方向移动会“去噪”图像。这个方向是对对数概率的正确的更新。即以上公式已经证明，学习对分数函数进行建模等价于建模源噪声的赋值（相对于一个比例因子）。"""

    """- 综上，我们推导了三个等效的目标来优化VDM:1.学习一个神经网络来预测原始图像$x_0$、源噪声$\epsilon_0$、或图像在任意噪声水平下的分数$\\nabla{logp(x_t)}$。该VDM可以通过随机采样时间步长t和利用地面真实目标最小化预测的范数来进行可伸缩性的训练。"""

    """$\\textbf{4.基于分数的生成模型}$"""
    """- 我们已经证明，一个变分扩散模型可以简单地通过优化一个 神经网络$s_{\\theta}(x_t,t)$来预测分数函数$\\nabla{logp(x_t)}$。但是，分数术语是来自于Tweedie公式的应用，这没有解释为什么它值得建模。因此，我们找到基于分数的生成模型来证明，之前推导的VDM公式有等价的基于分数的生成建模公式，而且可以随意切换。"""
    """- 1.为什么优化分数函数有意义？"""
    """ 首先，我们先看基于能量的模型。任意灵活的概率分布可写为："""
    st.latex(r"""
    p_{\theta}(x)=\frac{1}{Z_{\theta}}e^{-f_{\theta}(x)}
    """)
    """$ \qquad $ 其中，$f_{\\theta}$是一个任意灵活的、可以参数化的函数，称为能量函数，通常由神经网络建模，以确保$\int{p_{\\theta}(x)dx}=1$。学习这种分布的一种方法是最大似然，但是需要计算正则化常数$Z_{\\theta}=\int{e^{-f_{\\theta}(x)dx}dx}$，这对于复杂的$f_{\\theta}(x)$是不可能的。"""
    """$ \qquad $ 因此，可以使用神经网络$s_{\\theta}(x)$来学习分布p(x)的分数函数$\\nabla{logp(x)}$。这个想法来自于："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片134.png', caption='图134')
    """$ \qquad $ 它可以自由地表示为神经网络，没有任何正则化常数。分数模型可以通过使用地面真实分数函数最小化Fisher散度来进行优化："""
    st.latex(r"""
    E_{p(x)}[\Vert s_{\theta}(x)-\nabla{logp(x)} \Vert_2^2]
    """)

    """- 2.分数函数代表什么？"""
    """ 对于每一个x，取它的对数似然的梯度，本质上描述了在数据空间中移动的方向，以进一步增加其可能性。"""
    """ 直观地说，分数函数在数据x所在的整个空间上定义了一个向量场，指向modes。从视觉上看，如图所示："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片135.png', caption='图135')
    """$ \qquad $ 通过学习真实数据分布的得分函数，我们可以从同一空间中的任意一点开始，迭代地跟踪得分，知道达到一个 mode。这个采样过程被称之为朗之万动力学，在数学上被描述为："""
    st.latex(r"""
    x_{i+1} \leftarrow{x_i+c\nabla{logp(x_i)}+\sqrt{2c}\epslon}, i=1,1,..,K
    """)
    """$ \qquad $ 其中，$x_0$是从先验分布（如均匀）分布中随机采样的，$\epsilon \sim{N(\epsilon;0,I)}$是一个额外的噪声项，以确保生成的样本不总是崩溃到一个模式，而是在其周围盘旋以获得多样性。"""
    """$ \qquad $ 此外，由于学习到的分数函数是确定性的，因此涉及到噪声项的抽样增加了生成过程的随机性，允许我们避免确定性的轨迹。当从位于多个模式之间的未知初始化采样时，这点特别有用。朗之万动力学采样和噪声项的好处可视化如上图所示。"""
    """ 如上所示，优化目标依赖于对地面真实得分函数的访问，但是对于很多复杂的分布我们并不知道。但是，一些转换技术比如分数匹配可以在不知道地面真实分数的情况下，可以通过随机梯度下降进行优化。"""

    """ $\\textbf{总之，学习一个将分布表示为一个分数函数，并使用它通过马尔科夫链蒙特卡洛技术生成样本如朗之万动力学，被称为基于分数的生成建模。}$"""
    """$ \qquad $ """
    """$ \qquad $ """









with tab2:
    """$ \qquad $ Diffusion模型的数据生成过程带有十分朴素的思想，一个信号（图片 音频等）从某一个分布中采样后，经过无数次添加高斯噪声，最终能成为一个服从$N(0,I)$的真正的高斯分布，即$x_T\sim{N(0,I)}$，同时我们假设$x_{t-1}$的分布可以从$x_t$中得到，他们之前关系如下"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片76.png', caption='图76')
    """$ \qquad $ 同时我们假设x的采样过程为:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片77.webp', caption='图77')
    """$ \qquad $ $x_t$相对$x_{t-1}$的条件分布为由$\\beta_t$参数化的高斯分布:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片78.png', caption='图78')
    """$ \qquad $ 我们希望进行极大似自然估计$MLE(\\theta;p_{\\theta}(x_0))$，也就是最小化:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片79.png', caption='图79')
    """$ \qquad $ 通过Jensen不等式 我们可以得到对convex function $E[f(X)]\ge{f(EX)}$"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片80.png', caption='图80')
    """$ \qquad $ 这一形式和VAE中的变分下界是等价的，只不过下面的式子中的变分下界的z被替换成了$x_{1:T}$"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片81.png', caption='图81')
    """$ \qquad $ 为了让$L_{MLE}$的上界$L_{VLB}$更容易被优化我们可以对其进行进一步的整理，写成条件概率KL散度的形式"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片82.png', caption='图82')
    """$ \qquad $ 为了求解$L_{VLB}$首先我们需要推导$q(x_t|x_0)$"""
    """$ \qquad $ 令$\\alpha_t=1-\\beta_t\\bar{\\alpha}_t=\Pi_{i=1}^T{\\alpha_i}$"""
    """$ \qquad $ 我们已知$q(x_t|x_{t-1})=N(x_t;\sqrt{1-\\beta_t}x_{t-1},\\beta{I})$"""
    """$ \qquad $ 对条件概率进行重参数化得到"""
    st.latex(r"""x_t=\sqrt{\alpha}_tx_{t-1}+\sqrt{1-\alpha_t}z_{t-1}""")
    """$ \qquad $ z是从$N(0,I)$中采样得到的,因此"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片83.png', caption='图83')
    """$ \qquad $ 可以得到条件分布"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片84.png', caption='图84')
    """$ \qquad $ 接下来，推导可得"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片85.png', caption='图85')
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image('./pages/图片/图片86.png', caption='图86')
    with col2:
        st.image('./pages/图片/图片87.png', caption='图87')
    """$ \qquad $ 其中,"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片88.png', caption='图88')
    """$ \qquad $ 如果写成高斯分布的形式,可以得到"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片89.png', caption='图89')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片90.png', caption='图90')
    """$ \qquad $ 的每一项都可以写成关于的函数，此时可以通过从中采样x来实现损失函数的计算。"""

with tab3:
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片91.png', caption='图91')
    """$ \qquad $ 其中$L_T$因为不带有任何科学系的参数，是常数可以忽略不计。"""
    """$ \qquad $ 对于$L_i,i=1,...,T-1$，这一损失函数的形式过于复杂，首先改变$p_{\\theta}(x_{t-1}|x_t)$的建模方式,变成:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片92.png', caption='图92')
    """$ \qquad $ 因为对于两个多维高斯来说KL散度为:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片93.png', caption='图93')
    """$ \qquad $ 令$\sum_{\\theta}=\sigma_t^2I$,我们可以简化损失函数为"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片94.png', caption='图94')
    """$ \qquad $ 从而得到"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片95.png', caption='图95')
    """$ \qquad $ 最后训练过程变成了:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片96.png', caption='图96')
