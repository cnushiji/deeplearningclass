import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tab1, tab2, tab3, tab4, tab5 = st.tabs(['1.人工智能AI', '2.机器学习ML', '3.深度学习DL', '4.应用场景','5.发展现状'])

with tab1:
    """- 什么是人工智能？"""
    r"""人工智能是研究人类智能行为规律（如学习、计算、推理、思考、规划等），构造具有一定智慧能力的人工系统，以完成往常需要人的智慧才能胜任的工作。
    $ \quad $ —温斯顿 MIT"""
    """- 领域导图："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片9.png', caption='图9')
    """- AI三次发展浪潮："""
    r"""$ \qquad $ 1956年，达特茅斯会议，“人工智能”概念首次提出，历经60余年，人工智能经历了三次发展浪潮。
"""
    r"""$ \qquad $ 1956-1974：概念首次提出，该时期的核心是让机器具备逻辑推理能力，开发出计算机可以解决代数应用题、学习和使用英语的程序等"""

    r"""$ \qquad $ 1980-1987：解决特定领域问题的“专家系统”AI程序，知识库系统和知识工程是主要研究方向。"""

    r"""$ \qquad $ 1993-2011：1997年深蓝战胜国际象棋世界冠军、计算性能上的基础性障碍逐渐被克服。2006年，Hinton等人提出深度学习，是第三次浪潮的标志。
"""
    _, col1, _ = st.columns([1, 30, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片10.png', caption='图10')
    """- AI的两种划分："""
    r"""$ \qquad $ 第一种划分：中科院院士张钹、李德毅等认为人工智能的理论发展可划分为计算、感知、认知三个阶段。计算智能即快速计算、记忆存储能力；感知智能，即视觉、听觉、触觉等感知能力；认知智能即具有推理、可解释性的能力。
"""

    r"""$ \qquad $ 第二种划分：中科院院士徐波等认为，根据人工智能解决问题的不同阶段，可将其划分为感知智能、认知智能和决策智能三个阶段。随着及其认知水平的提高，越来越多的问题会交给机器决策，为了提高机器决策的准确度，需要加强复杂问题下，提升人机信任度，增强人类与智能系统交互协作智能的研究，即决策智能。
    """
    _, col1, _ = st.columns([1, 20, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片11.png', caption='图11')
    """- AI发展脉络："""
    _, col1, _ = st.columns([1, 16, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片12.png', caption='图12')

with tab2:
    """- 机器学习简介："""
    r"""$ \qquad $ 机器学习是利用计算机模拟人的学习能力，从样本数据中学习得到知识和经验，然后用于实际的推断和决策。"""

    r"""$ \qquad $ 机器学习是现阶段解决很多人工智能问题的主流方法，是现代人工智能的本质，目前正处于高速发展中。机器学习领域的经典理论，包括PAC学习理论、决策树、支持向量机SVM、Adaboost、神经网络、流形学习、随机森林等，并走向实用。
"""
    """- 回顾："""
    r"""$ \qquad $ 1950年图灵在图灵测试的文章中提及机器学习的概念。"""

    r"""$ \qquad $ 1952年，IBM的亚瑟·塞缪尔设计了一款可以学习的西洋跳棋程序，并在1956年正式提出了“机器学习”的概念。
"""
    r"""$ \qquad $ 如今普遍认为机器学习的算法是主要通过找出数据里隐藏的模式进而做出预测的识别模式，它是人工智能的一个重要子领域，而人工智能又与更广泛的数据挖掘（Data Mining）和知识发现（Knowledge Discovery in Database) 领域相交叉。
"""
    """- 相互关系："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片13.png', caption='图13')

    """- 基本过程："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片14.png', caption='图14')

    """- 研究进展："""
    r"""$ \qquad $ 50年代，图灵测试提出，塞缪尔开发西洋跳棋游戏。"""
    r"""$ \qquad $ 60年代末到70年代末，发展几乎停滞。"""
    r"""$ \qquad $ 80年代，使用神经网络反向传播算法训练多参数线性规划MLP。"""
    r"""$ \qquad $ 1986年，昆兰提出决策树算法（ID3算法）。"""
    r"""$ \qquad $ 90年，SVM算法逐渐得到快速发展和应用。"""
    r"""$ \qquad $ 2006年，Hinton提出深度学习。"""
    r"""$ \qquad $ 近十年来，机器学习研究热度高涨，主要集中在深度学习。"""

with tab3:
    """- 深度学习简介："""
    r"""$ \qquad $ 深度学习的模型，将原始数据映射到特征空间，克服了传统及其学习依赖人工特征的缺陷，是实现端到端学习的关键。深度学习的出现，让针对图像、语音、文本等复杂数据类型的感知类问题取得了真正意义上的突破，将人工智能推进到一个新时代。
"""
    """- 起源："""
    r"""$ \qquad $ MLP：1958年，心理学家Frank发明了感知机Perceptron。"""
    r"""$ \qquad $ MLP缺陷：Minsky和Papert发现了感知机的缺陷，其不能处理异或回路问题(XOR问题)，以及当时存在计算能力不足以处理大型神经网络的问题，于是整个神经网络的研究进入停滞期。
"""
    r"""$ \qquad $ Hinton：2006年，Hinton提出了神经网络深度学习算法，使神经网络的能力大大提高。"""

    """- 人工神经网络(ANN)："""
    r"""$ \qquad $ ANN是由大量处理单元互联组成的非线性、自适应信息处理系统。它是一种模仿动物神经网络行为特征，进行分布式并行信息处理的算法数学模型。
"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片15.png', caption='图15')
    """- 典型的ANN结构："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片16.png', caption='图16')
    r"""$ \qquad $ 特点： 1.非线性 $ \qquad $ 2.非常定性 $ \qquad $ 3.非局限性 $ \qquad $ 4.非凸性"""
    r"""$ \qquad $ 优点： 1.具有自学习能力 $ \qquad $ 2. 具有联想存储功能 $ \qquad $ 3.具有高速寻找最优解的能力
"""
    """- 深度学习发展脉络："""
    r"""$ \qquad $ 1.前向网络 $ \qquad $ 2.自学习自编码 $ \qquad $ 3.自循环神经网络 $ \qquad $ 4.强化学习"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片17.png', caption='图17')
    """- 前向网络："""
    r"""$ \qquad $ 日本学者邦彦提出神经认知机(Neocognitron)，给出了卷积和池化的思想。"""
    r"""$ \qquad $ 1986年，Hinton提出了反向传播训练MLP解决了感知机不能处理非线性学习的问题。"""
    r"""$ \qquad $ 1998年，Yann LeCun提出了一个七层的卷积神经网络LeNet-5以识别手写数字。"""
    r"""$ \qquad $ 2012年，Hinton研究组提出AlexNet在ImageNet上以巨大优势夺冠，引发深度学习热潮。"""
    r"""$ \qquad $ 2016年，何凯明提出残差网络ResNet,极大增加了网络深度，效果有很大提升，成为图像识别、目标检测网络总的骨干架构。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片18.png', caption='图18 CNN')

    """- 自学习自编码学习："""
    r"""$ \qquad $ 自学习自编码学习以生成模型为主，机器学习中生成模型一直占据着一个非常重要的地位，但基于神经网络的生成模型一直没有引起关注。"""
    r"""$ \qquad $ RBM：2006年，Hinton基于首先玻尔兹曼机RBM（20世纪80年代左右提出的基于无向图模型的能量物理模型）设计了一个机器学习的生成模型，并且将其堆栈成为深度信念网络（Deep Belief Network），使用逐层贪婪的方法进行训练，当时的效果不大理想。"""
    r"""$ \qquad $ 自编码器：自编码器（Auto-Encoder）也是Hinton等于80年代提出的模型，后来随着计算能力的提升重新登上舞台。"""
    r"""$ \qquad $ 变分自编码器：2014年，Welling等人使用神经网络训练一个有一层隐变量的图模型，称为变分自编码器。"""
    r"""$ \qquad $ 生成对抗模型：2014年，Goodfellow等提出生成对抗模型GAN，它通过判别器和生成器进行对抗训练生成模型。后续有大量跟进研究，包括DCGAN、WGAN等。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片19.png', caption='图19 Auto-Encoder')
    """- 自循环神经网络："""
    r"""$ \qquad $ 自循环神经网络模型，或者序列模型不是因为深度学习才有的，而是很早以前就有相关研究，例如有向图模型中的隐马尔科夫HMM、无向图模型中的条件随机场模型CRF等都是非常成功的序列模型。
"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片20.png', caption='图20 RNN模型')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片21.png', caption='图21 GAN')
    """- 增强学习(RL)："""
    r"""$ \qquad $ 增强学习（Rinforcement Learning），以强化学习为代表，它讨论的是一个智能体（Agent）怎么在一个复杂不确定的环境（environment）里面去极大化它能获得的奖励（Reward)。这个领域最出名的属谷歌的DeepMind公司。"""
    r"""$ \qquad $ 1.Q-learning是很有名的传统RL算法，Deep Q-learning将原来的Q值表用神经网络来代替。"""
    r"""$ \qquad $ 2.Double Dueling对这个思路进行一些扩展，主要是Q-learning的权重更新时序上。"""
    r"""$ \qquad $ 3.DeepMind的其它工作如DDPG、A3C也很有名，他们是基于策略梯度（Poicy Gradient）和神经网络结合的变种。"""
    r"""$ \qquad $ 4.AlphaGo，既用了RL的方法也有传统的蒙特卡洛搜索技巧，DeepMind后来提出了一个用AlphaGo框架，但通过自主学习来万不同游戏的新算法AlphaZero。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片22.png', caption='图22 RL代表算法')

    """- 深度学习三要素："""
    r"""$ \qquad $ 数据：数据是深度学习算法的粮食，很大程度上决定了算法性能的好坏，通常深度学习都需要大量数据来训练模型。"""
    r"""$ \qquad $ 算法：深度学习算法是深度学习在各种应用场景应用的关键，比如图像识别、目标检测、语音识别等等都需要相应的算法支持。"""
    r"""$ \qquad $ 算力：对图像、语音、文本等的海量数据，深度学习算法需要进行大量计算进行训练，因此能提供高算力支持的计算设备对深度学习任务来说必不可少。"""

with tab4:
    """- 智慧医疗："""
    r"""$ \qquad $ 1.影像诊断：计算机视觉、计算机图形、深度学习"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片23.png', caption='图23 新冠肺炎肺部CT图像智能诊断')
    r"""$ \qquad $ 2.医疗机器人：机器人、人机交互"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片24.jpg', caption='图24 假肢机器人')

    r"""$ \qquad $ 3.远程诊断：知识图谱、深度学习算法"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片25.jpg', caption='图25 远程脑起搏器成功实施，5G医疗落地')

    r"""$ \qquad $ 4.电子病历：自然语言处理、语音识别"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片26.jpeg', caption='图26 深度学习分析电子病历，预测患者病情发展')

    """- 智慧城市："""
    r"""$ \qquad $ 1.智慧物流及建筑服务系统：机器学习、物联网、安全与隐私"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片27.png', caption='图27 某物流公司的物流机器人正在高效工作')

    r"""$ \qquad $ 2.自动驾驶：机器学习、计算机视觉、人机交互"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片28.jpeg', caption='图28 自动驾驶汽车：通过电脑系统实现的无人驾驶的智能汽车')

    r"""$ \qquad $ 3.智能家居：自然语言处理、自然语言理解、安全与隐私、人机交互、语音识别、计算机视觉"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片29.jpg', caption='图29 人用平板电脑控制智能家居助手作为虚拟屏幕 ')

    r"""$ \qquad $ 4.智能交通系统：机器学习、数据挖掘、计算机视觉、自家语言处理、自然语言理解、图像识别、智能芯片"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片30.jpg', caption='图30 智能交通 ')

with tab5:
    """- 研究趋势："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter1_Introduction/pages/图片/图片31.png', caption='图31')

    """- AI的挑战："""
    r"""$ \qquad $ 1.模型的可解释性：目前的深度学习模型是黑盒模型，模型的结果不具有可解释性。"""
    r"""$ \qquad $ 2.模型的安全性、鲁棒性、可信性和扩展性：模型的结果是否鲁棒、安全可信、是否具有良好的更新重构和扩展性。"""
    r"""$ \qquad $ 3.如何解决安全和伦理方面的挑战：深度学习需要用到海量数据，确保数据的安全隐私，伦理方面如何保证不过渡依赖智能，如何消除算法歧视带来的偏见等非常重要。
"""