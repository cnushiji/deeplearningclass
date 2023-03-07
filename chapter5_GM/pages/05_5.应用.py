import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tab1, tab2 = st.tabs(['1.图片生成', '2.换脸'])

with tab1:
    """-  GAN生成数字"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片103.png', caption='图103')
    """-  主要用的库："""
    """ $ \qquad $ os模块用来对本地文件读写程序、查找等文件操作。"""
    """ $ \qquad $ numpy模块用来矩阵和数据的运算处理，其中也包括和深度学习框架之间的交互等。"""
    """ $ \qquad $ Keras是一个由Python编写的开源人工神经网络库，可以作为Tensorflow、Microsoft-CNTK和Theano的高阶程序接口，进行深度学习模型的设计、调试、评估、应用和可视化。"""
    """ $ \qquad $ Matplotlib模块用来可视化训练效果等数据图的操作。"""
    """- 模型初始化："""
    with st.expander('code'):
        st.code("""
        def __init__(self, width=28, height=28, channels=1):
            self.width = width
            self.height = height
            self.channels = channels
            self.shape = (self.width, self.height, self.channels)
            self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
            self.G = self.__generator()
            self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)
            self.D = self.__discriminator()
            self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
            self.stacked_generator_discriminator = self.__stacked_generator_discriminator()
            self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        """)
    """- 生成器模型的搭建："""
    """ $ \qquad $ 简单地搭建一个生成器模型：3个全连接层，使用sequentiao标准化。神经元数分别是256、512、1024等。"""
    with st.expander('code'):
        st.code("""
        def __generator(self):
            #Declare generator:
            model = Sequential()
            model.add(Dense(256, input_shape=(100,)))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(512))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(1024))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(self.width  * self.height * self.channels, activation='tanh'))
            model.add(Reshape((self.width, self.height, self.channels)))
            return model
        """)
    """- 判别器模型的搭建："""
    """ $ \qquad $ 同样简单搭建判别器网络层，和生成器模型类似。"""
    with st.expander('code'):
        st.code("""
        def __discriminator(self):
        #Declare discriminator
        model = Sequential()
        model.add(Flatten(input_shape=self.shape))
        model.add(Dense((self.width * self.height * self.channels), input_shape=self.shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.int64((self.width * self.height * self.channels)/2)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model
        """)
    """- 对抗式网络的搭建："""
    """ $ \qquad $ 在这里鉴别器的权重被冻结了，当我们训练这个模型时，生成器层将不受影响，只是向上传递梯度。"""
    with st.expander('code'):
        st.code("""
        def __stacked_generator_discriminator(self):
            self.D.trainable = False
            model = Sequential()
            model.add(self.G)
            model.add(self.D)
            return model
        """)
    """- 模型的训练："""
    with st.expander('code'):
        st.code("""
        def train(self, X_train, epochs=20000, batch = 32, save_interval = 100):
            for cnt in range(epochs):
                ## train discriminator
                random_index = np.random.randint(0, len(X_train) - np.int64(batch/2))
                legit_images = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels)
                gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100))
                syntetic_images = self.G.predict(gen_noise)
                x_combined_batch = np.concatenate((legit_images, syntetic_images))
                y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))
                d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)
                # train generator
                noise = np.random.normal(0, 1, (batch, 100))
                y_mislabled = np.ones((batch, 1))
                g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)
                print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))
                if cnt % save_interval == 0:
                    self.plot_images(save2file=True, step=cnt)
        """)
    """- 可视化："""
    with st.expander('code'):
        st.code("""
        def plot_images(self, save2file=False, samples=16, step=0):
            ''' Plot and generated images '''
            if not os.path.exists("./images"):
                os.makedirs("./images")
            filename = "./images/mnist_%d.png" % step
            noise = np.random.normal(0, 1, (samples, 100))
            images = self.G.predict(noise)
            plt.figure(figsize=(10, 10))
            for i in range(images.shape[0]):
                plt.subplot(4, 4, i+1)
                image = images[i, :, :, :]
                image = np.reshape(image, [self.height, self.width])
                plt.imshow(image, cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            if save2file:
                plt.savefig(filename)
                plt.close('all')
            else:
                plt.show()
        """)
    button = st.button('运行', key='MLP-mnist')
    if button:
        # -*- coding: utf-8 -*-
        import os
        import numpy as np
        from IPython.core.debugger import Tracer
        from keras.datasets import mnist
        from keras.layers import Input, Dense, Reshape, Flatten, Dropout
        from keras.layers import BatchNormalization
        from keras.layers.advanced_activations import LeakyReLU
        from keras.models import Sequential
        from keras.optimizers import Adam
        import matplotlib.pyplot as plt

        plt.switch_backend('agg')  # allows code to run without a system DISPLAY


        class GAN(object):
            """ Generative Adversarial Network class """

            def __init__(self, width=28, height=28, channels=1):
                self.width = width
                self.height = height
                self.channels = channels
                self.shape = (self.width, self.height, self.channels)
                self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)
                self.G = self.__generator()
                self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)
                self.D = self.__discriminator()
                self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
                self.stacked_generator_discriminator = self.__stacked_generator_discriminator()
                self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)

            def __generator(self):
                """ Declare generator """
                model = Sequential()
                model.add(Dense(256, input_shape=(100,)))
                model.add(LeakyReLU(alpha=0.2))
                model.add(BatchNormalization(momentum=0.8))
                model.add(Dense(512))
                model.add(LeakyReLU(alpha=0.2))
                model.add(BatchNormalization(momentum=0.8))
                model.add(Dense(1024))
                model.add(LeakyReLU(alpha=0.2))
                model.add(BatchNormalization(momentum=0.8))
                model.add(Dense(self.width * self.height * self.channels, activation='tanh'))
                model.add(Reshape((self.width, self.height, self.channels)))
                return model

            def __discriminator(self):
                """ Declare discriminator """
                model = Sequential()
                model.add(Flatten(input_shape=self.shape))
                model.add(Dense((self.width * self.height * self.channels), input_shape=self.shape))
                model.add(LeakyReLU(alpha=0.2))
                model.add(Dense(np.int64((self.width * self.height * self.channels) / 2)))
                model.add(LeakyReLU(alpha=0.2))
                model.add(Dense(1, activation='sigmoid'))
                model.summary()
                return model

            def __stacked_generator_discriminator(self):
                self.D.trainable = False
                model = Sequential()
                model.add(self.G)
                model.add(self.D)
                return model

            def train(self, X_train, epochs=20000, batch=32, save_interval=100):
                for cnt in range(epochs):
                    ## train discriminator
                    random_index = np.random.randint(0, len(X_train) - np.int64(batch / 2))
                    legit_images = X_train[random_index: random_index + np.int64(batch / 2)].reshape(
                        np.int64(batch / 2), self.width, self.height, self.channels)
                    gen_noise = np.random.normal(0, 1, (np.int64(batch / 2), 100))
                    syntetic_images = self.G.predict(gen_noise)
                    x_combined_batch = np.concatenate((legit_images, syntetic_images))
                    y_combined_batch = np.concatenate(
                        (np.ones((np.int64(batch / 2), 1)), np.zeros((np.int64(batch / 2), 1))))
                    d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)
                    # train generator
                    noise = np.random.normal(0, 1, (batch, 100))
                    y_mislabled = np.ones((batch, 1))
                    g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)
                    print(
                        'epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))
                    if cnt % save_interval == 0:
                        self.plot_images(save2file=True, step=cnt)

            def plot_images(self, save2file=False, samples=16, step=0):
                ''' Plot and generated images '''
                if not os.path.exists("./images"):
                    os.makedirs("./images")
                filename = "./images/mnist_%d.png" % step
                noise = np.random.normal(0, 1, (samples, 100))
                images = self.G.predict(noise)
                plt.figure(figsize=(10, 10))
                for i in range(images.shape[0]):
                    plt.subplot(4, 4, i + 1)
                    image = images[i, :, :, :]
                    image = np.reshape(image, [self.height, self.width])
                    plt.imshow(image, cmap='gray')
                    plt.axis('off')
                plt.tight_layout()
                if save2file:
                    plt.savefig(filename)
                    plt.close('all')
                else:
                    plt.show()


        if __name__ == '__main__':
            (X_train, _), (_, _) = mnist.load_data()
            # Rescale -1 to 1
            X_train = (X_train.astype(np.float32) - 127.5) / 127.5
            X_train = np.expand_dims(X_train, axis=3)
            gan = GAN()
            gan.train(X_train)

    """ $ \qquad $ """
with tab2:
    """- $\\textbf{Face2Face: Real-time Face Capture and Reenactment of RGB Videos}$"""
    """- 目标：将由RGB传感器捕获的源参与者的面部表情在线转移到目标参与者。在这之前的大都是离线转移方法"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片104.png', caption='图104')
    """- 贡献：实时monocular面部再现。"""
    """- 面部再现的意义：它是各种应用的基础，如在视频会议中，源视频可以被调整为匹配翻译的面部动作，或者面部视频可以配音成外语等等。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片105.png', caption='图105 方法概述')
    """- 1.密集、全局和非刚性的基于模型的绑定"""
    """- 2.在无约束的实时RGB视频中的精确的跟踪、外观和 lighting估计"""
    """- 3.使用用子空间变形的依赖于人的表达转移"""
    """- 4.一种新的口腔合成方法"""
    """ 1.面部图像的合成"""
    """ 文章使用一个多线性PCA模型，前2个维度代表面部特征即几何形状和皮肤反射率，第3个维度控制面部表情。因此，将1张脸参数化为："""
    st.latex("""
    M_{geo}(\\alpha,\delta)=a_{id}+E_{id}\cdot{\\alpha}+E_{exp}\cdot{\delta},\qquad
    M_{alb}(\\beta)=a_{alb}+E_{alb}\cdot{\\beta}.
    """)
    """ 该先验假设形状和反射率的多元正太概率分布围绕：平均形状$a_{id}\in{R^{3n}}$和反射率$a_{alb}\in{R^{3n}}$。"""
    """ 给出了形状$E_{id}\in{R^{3n\\times{80}}}$，反射率$E_{alb}\in{R^{3n\\times{80}}}$，表情$E_{exp}\in{R^{3n\\times{76}}}$基和相应的标准差$\sigma_{id}\in{R^{80}},\sigma_{alb}\in{R^{80}},
    \sigma_{exp}\in{R^{76}}$。"""
    """ 模型有53K个顶点和106K个面。通过一个刚性模型变换$\Phi{v}$和全视角变换$\Pi{v}$下的模型 rasterization，生成合成图像$C_S$。"""

    """ 合成依赖于人脸模型参数$\\alpha,\\beta,\delta$，照明参数$\gamma$，刚性变换R、t,和定义$\Phi$的相机参数k。未知P的向量是这些参数的并集。"""

    """ 2.能量公式"""
    """ 给定一个monocular输入序列，使用一个变分优化重构所有未知参数P。所提出的目标是高度非线性的，具有以下组成部分："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片106.png', caption='图106')
    """ 其中，数据项用photo一致性$E_{col}$和面部特征对齐$E_{lan}$来衡量合成图像和数据数据之间的相似性。统计正则化器$E_{reg}$考虑了给定参数向量P的可能性。权重$w_{col},w_{lan},w_{reg}$平衡了3个不同的子目标。"""
    """$ \qquad $ 子目标1——Photo一致性"""
    """$ \qquad $ 为了量化合成图像对输入数据的解释程度，我们测量了像素级上的photo-metric对齐误差："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片107.png', caption='图107')
    """$ \qquad $ 其中，$C_S$是合成图像，$C_I$是输入的RGB图像，$p\in{V}$表示$C_S$中所有可见的像素位置。
    使用$l_{2,1}-$范数而不是最小二乘公式来对异常值具有鲁棒性。在场景中，颜色空间中的距离是基于$l_2$的;而在所有像素的求和中，使用一个$l_1-$范数来加强稀疏性。"""

    """$ \qquad $ 子目标2——特征对齐"""
    """$ \qquad $ 文章还加强了在RGB流中检测到的一组显著的面部特征点对之间的特征相似性："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片108.png', caption='图108')
    """$ \qquad $ 文章采用了面部地标跟踪算法。每个特征点$f_j\in{F}\subset{R^2}$都有一个检测置信度$w_{conf,j}$，并对应于我们的面部先验的唯一顶点$v_j=M_{geo}(\\alpha,\delta)\in{R^3}$,这有助于避免$E_{col}(p)$在高度复杂的能量landscape中的局部最小值。"""

    """$ \qquad $ 子目标3——统计正则化"""
    """$ \qquad $ 基于一个正太分布的总体假设，加强了合成的faces的合理性。为此，强制这些参数在统计学上接近于平均值："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片109.png', caption='图109')
    """$ \qquad $ 这种常用的正则化策略可以防止人脸几何形状和反射率的退化，并指导局部最小的优化策略。"""

    """ 3.数据并行优化策略"""
    """$ \qquad $ 所提出的鲁棒跟踪目标是一个一般的无约束非线性优化问题。使用一种新的基于数据并行gpu的Iteratively Reweighted Least Squares（IRLS）求解器来实时最小化这个目标。IRLS的关键思想是，在每次迭代中，通过将范数分成两个分量，将问题转化为一个非线性最小二乘问题："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片110.png', caption='图110')
    """$ \qquad $ 其中，$r(\cdot)$是一个一般的残差，$P_{old}$是在最后一次迭代中计算出的解。每单个迭代步都是使用Gauss-Newton方法实现的。
    在每次IRLS迭代中采用单步GN，基于PCG求解相应的正规方程组$J^TJ\delta^{*}=-J^TF$，得到最优线性参数更新$\delta^{*}$。"""

    """ 4.非刚性模型的绑定"""
    """$ \qquad $ 为了估计在monocular重建的欠约束的情况下参与者的身份，文章引入了一种非刚性模型的捆绑方法。基于提出的目标，文章联合估计了输入视频序列的k个关键帧上的所有参数。
    估计的未知参数是全局恒等$\left\{\\alpha,\\beta\\right\}$和内部的k,以及未知的每帧的姿态
    $\left\{\delta^k,R^k,t^k \\right\}_k$和照明参数$\left\{\gamma^k \\right\}_k$。我们使用类似的数据并行优化策略来进行模型到帧的跟踪，但联合求解整个关键帧集的正常方程。
    对于基于非刚性模型的捆绑问题，相应的雅可比矩阵的非零结构是块密集的。我们的PCG求解器利用非零结构来提高性能。
    由于所有关键帧在潜在变化的照明、表达式和视角下观察到相同的面部身份，因此我们可以稳健地将身份从所有其他问题维度中分离出来。注意，我们还解决了$\Pi$的固有相机参数，因此能够处理未校准的视频片段。"""

    """ 5.Expression Transfer"""
    """ 为了将表达式的变化从源传递到目标行动者，同时保留每个行动者的表达式中的人的特征，我们提出了一种子空间变形传递技术。为了将表达式的变化从源传递到目标行动者，同时保留每个行动者的表达式中的人的特征，我们提出了一种子空间变形传递技术。"""
    """ 假设源identity $\\alpha^S$和$\\alpha^T$固定，转移以neutral $\delta^S_N$，变形源$\delta^S$，和neutral 目标$\delta^T_N$表情作为输入。输出是直接在参数先验的简化子空间中转移的面部表情$\delta^T$。"""
    """ 根据之前提出的方法，首先计算了源变形的梯度$A_I\in{R^{3\\times{3}}}$，即将源三角形从neutral到变形状态。基于未变形状态$v_i=M_i(\\alpha^T,\delta^T_N)$，通过求解一个线性最小二乘问题，找到变形的目标$\hat{v}_i=M_i(\\alpha^T,\delta^T)$。设$(i_0,i_1,i_2)$是第i个三角形的顶点指数，
    $V=[v_{i_1}-v_{i_0},v_{i_2}-v_{i_0}],\hat{V}=[v_{i_1}-\hat{v}_{i_0},v_{i_2}-\hat{v}_{i_0}]$，则最优的未知目标变形$\delta^T$为如下公式的最小值："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片111.png', caption='图111')
    """ 可以重写为标准的最小二乘形式："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片112.png', caption='图112')
    """$ \qquad $ 其中，矩阵$A\in{R^{6|F|\\times{76}}}$是常数，包含投影到表达式子空间的模板网格的边缘信息。在中性表情中目标的边缘信息包括在右手边的$b\in{R^{f|F|}}$。b随着$\delta^S$的变化而变化，并在每个新的输入帧的GPU上计算。
    二次能量的正态方程可以求出最小值。由于系统矩阵是常数的，我们可以用奇异值分解（SVD）来预先计算它的伪逆。
    随后，实时求解了小型的76×76线性系统。因为弯曲形状模型隐式地将结果限制在可信的形状，并保证平滑性。"""

    """ 6.口型检索"""
    """ 对于一个给定的转移的面部表情，我们需要合成一个真实的目标口型区域。为此，我们从目标参与者序列中检索和扭曲最佳匹配的嘴图像。我们假设在目标视频中有足够的口型变化。同样值得注意的是，我们要保持目标嘴的外观。"""
    """ 我们的方法是用一个新的特征相似性度量，基于一个框架到聚类策略，首先找到最合适的目标嘴帧。"""
    """ 为了加强时间一致性，我们使用一个密集的外观图来找到最后检索到的嘴帧和目标嘴帧之间的折衷方案。如图："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片113.png', caption='图113')
    """$ \qquad $ 1.相似性度量"""
    """$ \qquad $ 文章的相似度度量基于几何和photometric 特征。一个帧的$K=\left{ R,\delta,F,L \\right}$是由旋转R，表情参数$\delta$，lansmarks F和局部二进制模式(LBP)L组成。"""
    """$ \qquad $ 我们为训练序列中的每一帧计算这些参数$K^S$。目标描述符$K^T$由表情转移的结果和驱动参与者的帧的LBP组成。我们测量源和目标参数之间的距离如下："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片114.png', caption='图114')
    """$ \qquad $ 第一项Dp度量参数空间中的距离："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片115.png', caption='图115')
    """$ \qquad $ 第二项$D_m$测量稀疏面部landmarks的差异相容性："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片116.png', caption='图116')
    """$ \qquad $ 在这里，$\Omega$是一组预定义的landmark对，定义了诸如上唇和下唇之间或口腔左右角之间的距离。最后一项$D_a$是一个外观测量项，由以下两部分组成："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter5_GM/pages/图片/图片117.png', caption='图117')
    """$ \qquad $ $\\tau$是用于前一帧中再现的最后检索帧索引。$D_l(K^T,k^S_t)$基于通过Chi Squared距离进行比较的LBPs来衡量相似性。
    $D_c(\\tau,t)$基于归一化口帧的RGB互相关来度量最后检索到的帧$\\tau$与视频帧t之间的相似性。"""
    """$ \qquad $ 我们将这种帧到帧的距离度量应用到帧到集群匹配策略中，从而实现实时速率，并减轻口帧之间的高频跳跃。"""
    """$ \qquad $ 2.帧到cluster的匹配"""
    """$ \qquad $ 利用所提出的相似性度量，我们使用基于成对距离函数D的改进的k-means算法将目标参与者序列聚类成k = 10聚类。对于每个集群，我们选择与该集群内所有其他帧距离最小的帧作为代表。在运行时，我们测量目标描述符KT与集群代表描述符之间的距离，并选择代表帧距离最小的集群作为新的目标帧。"""

    """$ \qquad $ 3.外观graph"""
    """$ \qquad $ 我们通过建立所有视频帧的全连接外观图来改善时间相干性。边缘权值是基于归一化口帧之间的RGB互相关关系、参数空间$D_p$中的距离和地标$D_m$之间的距离。该图使我们能够找到一个与上一个检索到的帧和检索到的帧相似的目标帧之间的帧。我们通过找到训练序列的框架来计算这个完美的匹配，以最小化到最后一个检索到的和当前的目标框架的边缘权值之和。"""
    """$ \qquad $ 我们在optic flow alignment后的像素水平上的纹理空间中混合了先前检索到的框架和新检索到的框架"""
    """$ \qquad $ 在混合之前，我们应用了一个照明校正，它考虑了检索帧和当前视频帧的估计的球面谐波照明参数。最后，我们通过在原始视频帧、光照校正帧、投影的口帧和渲染的人脸模型之间的alpha混合来合成新的输出帧。"""