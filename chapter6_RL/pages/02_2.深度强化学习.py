import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tab1, tab2, tab3, tab4 = st.tabs(['1.基础知识','2.AlphaGo', '3.AlphaFold', '4.AlphaFold2'])

with tab1:
    """- 智能体（Agent）"""
    """$ \qquad $  感知外界环境的状态（State）和奖励反馈（Reward），并进行学习和决策。智能体的决策功能是指根据外界环境的状态来做出不同的动作（Action），而学习功能是指根据外界环境的奖励来调整策略。"""
    """- 环境（Environment）"""
    """$ \qquad $  智能体外部的所有事物，并受智能体动作的影响而改变其状态，并反馈给智能体相应的奖励。"""
    """- 强化学习中的基本要素:"""
    """$ \qquad $  环境的状态集合：S；"""
    """$ \qquad $  智能体的动作集合：A；"""
    """$ \qquad $  状态转移概率：$p(s^{’}|s,a)$，即智能体根据当前状态s做出一个动作a之后，下一个时刻环境处于不同状态$s^{’}$的概率；"""
    """$ \qquad $  即时奖励：$R : S\\times{A}\\times{S^{’}}\\to{R}$，即智能体根据当前状态做出一个动作之后，环境会反馈给智能体一个奖励，这个奖励和动作之后下一个时刻的状态有关。"""
    """$ \qquad $  强化学习问题可以描述为一个智能体从与环境的交互中不断学习以完成特定目标（比如取得最大奖励值）。"""
    """$ \qquad $  强化学习就是智能体不断与环境进行交互，并根据经验调整其策略来最大化其长远的所有奖励的累积值。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片14.png', caption='图14 ')
    """- 马尔可夫决策过程:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片15.png', caption='图15 ')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片16.png', caption='图16 ')
    """$ \qquad $ 马尔可夫过程:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片17.png', caption='图17 ')
    """- 策略$\phi(a|s)$:"""
    """$ \qquad $ 马尔可夫决策过程的一个轨迹（trajectory）:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片18.png', caption='图18 ')
    """$ \qquad $ τ的概率:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片19.png', caption='图19 ')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片20.png', caption='图20 ')
    """- 总回报:"""
    """$ \qquad $ 给定策略π(a|s)，智能体和环境一次交互过程的轨迹τ 所收到的累积奖励为总回报（return）"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片21.png', caption='图21 ')
    """$ \qquad $ $\gamma{\in{[0,1]}}$是折扣率。当$\gamma$接近于0时，智能体更在意短期回报；而当$\gamma$接近于1时，长期回报变得更重要。"""
    """$ \qquad $ 环境中有一个或多个特殊的终止状态（terminal state）。"""
    """- 强化学习目标函数:"""
    """$ \qquad $ 强化学习的目标是学习到一个策略$\pi_{\\theta}{(a|s)}$来最大化期望回报（expected return）。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片22.png', caption='图22 ')
    """$ \qquad $ $\\theta$为策略函数的参数。"""
    """- 强化学习的算法:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片23.png', caption='图23 ')
    """- 深度强化学习:"""
    """$ \qquad $ 深度强化学习是将强化学习和深度学习结合在一起，用强化学习来定义问题和优化目标，用深度学习来解决状态表示、策略表示等问题。"""
    """$ \qquad $ 两种不同的结合强化学习和深度学习的方式，分别用深度神经网络来建模强化学习中的值函数、策略，然后用误差反向传播算法来优化目标函数。"""

    """**基于值函数的策略学习**"""
    """- 如何评估策略$\pi_{\\theta}{(a|s)}$？"""
    """$ \qquad $ 两个值函数"""
    """$ \qquad $ 状态值函数"""
    """$ \qquad $ 状态-动作值函数"""

    """- 状态值函数:"""
    """$ \qquad $ 一个策略$\pi$期望回报可以分解为"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片24.png', caption='图24 ')
    """$ \qquad $ 值函数：从状态s开始，执行策略π得到的期望总回报"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片25.png', caption='图25 ')
    """- Bellman方程:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片26.png', caption='图26 ')
    """- 状态-动作值函数（ Q函数）:"""
    """$ \qquad $ 状态-动作值函数是指初始状态为s并进行动作a，然后执行策略π得到的期望总回报。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片27.png', caption='图27 ')
    """$ \qquad $ Q函数的贝尔曼方程"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片28.png', caption='图28 ')
    """- 最优策略:"""
    """$ \qquad $ 最优策略：存在一个最优的策略π∗ ，其在所有状态上的期望回报最大。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片29.png', caption='图29 ')
    """$ \qquad \qquad $ 难以实现"""
    """$ \qquad $ 策略改进："""
    """$ \qquad \qquad $ 值函数可以看作是对策略$\pi$的评估。"""
    """$ \qquad \qquad $ 如果在状态s，有一个动作a使得$Q^{\pi}{(s,a)} >V^{\pi}{(s)}$，说明执行动作a比当前的策略$\pi{(a|s)}$要好，我们就可以调整参数使得策略$\pi{(a|s)}$的概率增加。"""
    """- 如何计算值函数？"""
    """$ \qquad $ 基于模型的强化学习算法"""
    """$ \qquad \qquad $ 基于MDP过程：状态转移概率$p(s^{’}|s,a)$和奖励函数R(s,a,s^{’})。"""
    """$ \qquad \qquad $ 策略迭代"""
    """$ \qquad \qquad $ 值迭代"""
    """$ \qquad $ 模型无关的强化学习:"""
    """$ \qquad \qquad $ 无MDP过程"""
    """$ \qquad \qquad $ 蒙特卡罗采样方法"""
    """$ \qquad \qquad $ 时序差分学习"""

    """**基于模型的强化学习**"""
    """- 策略迭代:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片30.png', caption='图30 ')
    """- 值迭代:"""
    """$ \qquad $ 值迭代方法将策略评估和策略改进两个过程合并，来直接计算出最优策略。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片31.png', caption='图31 ')

    """**模型无关的强化学习:**"""
    """- 蒙特卡罗采样方法:"""
    """$ \qquad $ 策略学习过程:"""
    """$ \qquad \qquad $ 通过采样的方式来计算值函数，"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片32.png', caption='图32 ')
    """$ \qquad \qquad \qquad $ 当$N\\to{∞}$时，$\hat{𝑄}_\pi{(𝑠,𝑎)}\\to{𝑄_\pi{(𝑠,𝑎)}}$。"""
    """$ \qquad \qquad $ 在似估计出$Q_\pi{(s,a)}$之后，就可以进行策略改进。"""
    """$ \qquad \qquad $ 然后在新的策略下重新通过采样来估计Q函数，并不断重复，直至收敛。"""

    """- $\epsilon$-贪心法:"""
    """$ \qquad $ 利用和探索"""
    """$ \qquad \qquad $ 对当前策略的利用（Exploitation）,"""
    """$ \qquad \qquad $ 对环境的探索（Exploration）以找到更好的策略"""
    """$ \qquad $ 对于一个确定性策略$\pi$，其对应的$\epsilon$-贪心法策略为:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片33.png', caption='图33 ')
    """- 时序差分学习方法:"""
    """$ \qquad $ 结合了动态规划和蒙特卡罗方法"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片34.png', caption='图34 ')
    """$ \qquad $ 从s,a开始，采样下一步的状态和动作$(s^{′},a^{′})$，并得到奖励$r(s,a,s^{′})$，然后利用贝尔曼方程来近似估计G(τ)。"""
    """- SARSA算法（State Action Reward State Action，SARSA）:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片35.png', caption='图35 ')
    """- Q学习算法:"""
    """$ \qquad $ Q学习算法不通过$\pi^{\epsilon}$来选下一步的动作$a^{′}$ ，而是直接选最优的Q函数，"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片36.png', caption='图36 ')
    """- 基于值函数的深度强化学习:"""
    """$ \qquad $ 为了在连续的状态和动作空间中计算值函数$Q_{\pi}{(s,a)}$，我们可以用一个函数$Q_{\phi}{(s,a)}$来表示近似计算，称为值函数近似（Value Function Approximation）。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片37.png', caption='图37 ')
    """- 目标函数:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片38.png', caption='图38 ')
    """$ \qquad $ 存在两个问题："""
    """$ \qquad \qquad $ 目标不稳定，参数学习的目标依赖于参数本身；"""
    """$ \qquad \qquad $ 样本之间有很强的相关性。"""
    """$ \qquad $ 深度Q网络"""
    """$ \qquad \qquad $ 一是目标网络冻结（freezing target networks），即在一个时间段内固定目标中的参数，来稳定学习目标；"""
    """$ \qquad \qquad $ 二是经验回放（experience replay），构建一个经验池来去除数据相关性。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片39.png', caption='图39 ')
    """- DQN in Atari :"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片40.png', caption='图40 Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.')
    """- DQN in Atari :  Human Level Control:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片41.png', caption='图41 Mnih, Volodymyr, et al. 2015.')

    """**策略梯度:**"""
    """- 基于策略函数的深度强化学习:"""
    """$ \qquad $ 可以直接用深度神经网络来表示一个参数化的从状态空间到动作空间的映射函数：$a = \pi_{\\theta}{(s)}$。"""
    """$ \qquad $ 最优的策略是使得在每个状态的总回报最大的策略，因此策略搜索的目标函数为:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片42.png', caption='图42 ')
    """- 策略梯度（Policy Gradient）:"""
    """$ \qquad $ 策略搜索是通过寻找参数$\\theta$使得目标函数$J(\\theta)$最大。"""
    """$ \qquad $ 梯度上升："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片43.png', caption='图43 ')
    """- 进一步分解:"""
    """$ \qquad $ 参数$\\tau$优化的方向是使得总回报$G(\\tau)$越大的轨迹$\\tau$的概率$p_{\\theta}(\\tau)$也越大。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片44.png', caption='图44 ')
    """$ \qquad $ $\\frac{\partial{\\theta{logp_{\\theta}{\\tau}}}}{\partial{\\theta}}$是和状态转移概率无关，只和策略函数相关。"""
    """- 策略梯度:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片45.png', caption='图45 ')
    """- REINFORCE算法:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片46.png', caption='图46 ')
    """- 如何减少方差？"""
    """$ \qquad $ 训练方差"""
    """$ \qquad \qquad $ REINFORCE算法的一个主要缺点是不同路径之间的方差很大，导致训练不稳定，这是在高维空间中使用蒙特卡罗方法的通病。"""
    """$ \qquad $ 控制变量法"""
    """$ \qquad \qquad $ 假设要估计函数f 的期望，为了减少f 的方差，我们引入一个已知期望的函数g，令"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片47.png', caption='图47 ')
    """- 引入基线:"""
    """$ \qquad $ 在每个时刻t，其策略梯度为"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片48.png', caption='图48 ')
    """$ \qquad $ 为了减小策略梯度的方差，引入一个和$𝑎_𝑡$无关的基准函数$𝑏(𝑠_𝑡)$:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片49.png', caption='图49 ')
    """- 如何选择$𝑏(𝑠_𝑡)$ ？"""
    """$ \qquad $ 为了可以有效地减小方差， $𝑏(𝑠_𝑡)$和$𝐺(\\tau_(𝑡:𝑇))$越相关越好，一个很自然的选择是令$𝑏(𝑠_𝑡)$为值函数$𝑉^{\pi_{\\theta}}(𝑠_t)$."""
    """$ \qquad $ 一个可学习的函数$𝑉_{\phi}(𝑠_𝑡)$来近似值函数。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片50.png', caption='图50 ')
    """$ \qquad $ 策略函数参数θ的梯度为:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片51.png', caption='图51 ')
    """- 带基准线的REINFORCE算法:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片52.png', caption='图52 ')
    """- Actor-Critic算法:"""
    """$ \qquad $ 演员-评论员算法（Actor-Critic Algorithm）是一种结合策略梯度和时序差分学习的强化学习方法。"""
    """$ \qquad \qquad $ 演员（actor）是指策略函数𝜋_𝜃  (𝑠,𝑎)，即学习一个策略来得到尽量高的回报。"""
    """$ \qquad \qquad $ 评论员（critic）是指值函数𝑉_𝜙 (𝑠) ，对当前策略的值函数进行估计，即评估actor的好坏。"""
    """$ \qquad $ 借助于值函数，Actor-Critic算法可以进行单步更新参数，不需要等到回合结束才进行更新。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片53.png', caption='图53 ')
    """- 不同强化学习算法之间的关系:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片54.png', caption='图54 ')
    """- 汇总:"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片55.png', caption='图55 ')

with tab2:
    """$ \qquad $ AlphaGo使用两个不同的神经网络“大脑”，通过两者合作得出移棋决定。这些大脑是多层的神经网络，与之前的图像搜索引擎，比谷歌图片搜索引擎中的神经网络相比，结构上几乎相同。就像一个图像分类网络处理图像一样，他们从2D过滤器的几个等级层级开始，使过滤器处理围棋棋盘的位置。粗略地说，第一批过滤辨认图案和形状。这层过滤后，13个全连接的神经网络层输出它们对所看到的位置的判断。大概就是，这些层进行的是分类或逻辑推理。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片4.jpg', caption='图4')
    """$ \qquad $ 通过反复检查自己的结果，反馈那些通过调整数字使网络性能更好的修正，从而更好地训练网络。这个过程具有大量的随机性元素，所以不可能确切地知道网络如何“思想”，只能知道它是经过训练来提高的。"""
    """- 脑1：移动选择器:"""
    """$ \qquad $ AlphaGo的第一个神经网络的大脑，在论文中被称为“监督学习（SL）政策网络”，它着眼于棋盘中的位置，并试图决定最佳的下一步。实际上，它用来估计每个合法下一步行动为最好的一步的可能性，其顶层猜测就是具有最高概率的那步。你可以认为这是一个关于“移动选择”的大脑。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片5.webp', caption='图5')
    """$ \qquad $ 移动选择器如何看到棋盘。数字表示一个厉害的人类玩家把自己的下一步棋下到相应位置的相对可能性。图片来自Silver 等人。"""
    """$ \qquad $ Silver的团队通过几百万个优秀的人类棋手在KGS上的下棋选择，训练这个大脑。这是AlphaGo的最像人类的一部分，其目的仅仅是复制优秀的人类棋手的移动选择。它一点也不关心赢得比赛，只下那步顶级人类棋手会下的那步棋。 AlphaGo的下棋选择器有57％的概率可以正确匹配优秀的棋手下棋选择。 （错配不一定是错误 – 这可能是下错这步棋的人类的错误！）。"""
    """$ \qquad $ **下棋选择器，更强**"""
    """$ \qquad $ 整个AlphaGo系统实际上需要另外两个基本的执行下棋选择的版本。其中一个版本，“增强学习（RL）的政策网络”，是经几百万个附加的模拟游戏的密集型训练后的改良版本。可把这个作为“更强”的移动选择器。与上面描述的基本训练相比，上面的只教网络模仿一个人的下棋选择，高级培训完成每个模拟游戏直到结束，以教会网络哪些步骤最可能导致最终获胜。Silver的研究小组通过让更强的移动选择器与早期训练迭代中的版本对打，从而综合了几百万个训练游戏结果。"""
    """$ \qquad $ 单是厉害的移动选择器已经是一个强大的棋手，它处于业余低段级范围，而且与之前最强的围棋AI相比水平相当。而需要注意的是，这一移动选取器没有“阅读”的说法。它只是审视一个棋盘上的位置，并提出基于位置分析的下一步。它并不试图模拟未来任何的一步棋。这体现了简单的深层神经网络技术的惊人力量。"""
    """$ \qquad $ **移动选择器，更快**"""
    """$ \qquad $ 当然，该AlphaGo队并没有止步于此。一会儿我将描述他们是如何增加AI的阅读能力的。为了让阅读能力可以实现，他们需要一个更快版本的移动选择大脑。但更强大的版本输出下棋选择的时间太长——虽然对于下一步好棋来说速度已经很快，但阅读计算需要在做出决定之前检查数以千计的可能性。"""
    """$ \qquad $ Silver的团队建立了一个简化的移动选择器以创建一个“快速阅读”的版本，他们称之为“网络部署”。简化版本不观察整个19×19板，而只是看对手上一步棋附近的小窗，和考虑下一步棋走的位置。虽然取下部分移动选取器的大脑，但是精简的版本计算起来快1000倍，这样更适合阅读计算。"""

    """- 脑2：位置评估器:"""
    """$ \qquad $ AlphaGo的第二个大脑与移动选择器功能不同。它不猜测具体的下一步怎么走，而是通过设想的棋盘分布，估计每个玩家赢得比赛的概率。这个“位置评估”，在论文中被称为“价值网络”，通过提供整体的位置判断来配合移动选择器。这个判断只是近似的，但它对加快阅读速度非常有用。通过将未来可能的位置分为“好”或“坏”的分类，AlphaGo可以决定是否要沿着一个特定的变化进行更深的阅读。如果位置评估器说某个具体的变化看起来情况不妙，那么AI可以跳过阅读，不沿着那条线继续发挥。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片6.webp', caption='图6')
    """$ \qquad $ 位置评估器怎样看棋盘。深蓝色代表对棋手来说更可能导致赢棋的位置。图来自Silver等人。"""
    """$ \qquad $ 位置评估器经过了数以百万计的棋盘位置样例的训练。Silver的研究小组从AlphaGo的移动选择器的两个副本中进行的模拟游戏中选择了随机样本，然后创建了这些位置。要注意的是，对于构建一个足够大的数据集来训练位置评估，AI的移动选择器非常宝贵。移动选择器可以支持团队模拟从任何给定的棋盘位置上开始的可能的未来棋子位置的选择，从而猜测每个棋手的大致胜算。但是有可能世界上还没有足够的人类比赛记录可用来完成这种训练。"""
    """$ \qquad $ **增加阅读**"""
    """$ \qquad $ 三个版本的移动——选择的大脑，再加上位置——评估的大脑，AlphaGo现在可以有效阅读未来的棋子移动的序列。阅读也是通过Monte Carlo 搜索树算法（MCTS）实现的，这个算法被很多高级的围棋AI使用。但可以做出更多关于探索怎样的下棋变化，如何更深层地探索等问题的智慧性的猜测，使AlphaGo比其他AI更出色。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片7.webp', caption='图7 Monte Carlo （蒙特卡洛）树搜索算法。')
    """$ \qquad $ MCTS具有无限的计算能力，理论上通过探索游戏可能延续的每一种情况，它可以计算出最优的选择。然而，由于围棋游戏的未来移动选择的搜索空间如此之大（大于已知的宇宙粒子的数量），现实存在的AI没有可能探索各种可能的变化。能用MCTS运算，实际上取决于，AI在跳过探索坏的选择，识别未来变化的过程中有多出色。"""
    """$ \qquad $ Silver的团队将AlphaGo配置了模块化的MCTS系统。该框架允许设计师“插入”评估变化的不同功能。全功率AlphaGo系统以下列方式运作所有的“大脑”："""
    """$ \qquad $ 1.从目前的棋盘位置，选择几个可能会走的下一步棋。为做到这一点，他们使用“基本”的移动选择大脑。 （他们尝试过这一步使用“更强”的版本，但实际上这却会使AlphaGo性能变弱，因为它没有提供给MCTS系统提供足够大的变化空间来探索。而且这会过于看重“明显最好的”一步，而不是阅读那些可能会使自己在接下来的游戏中表现更好的几种选择。"""
    """$ \qquad $ 2.对于下一步棋每个可能的位置，可用以下的两种方法之一评估其质量：要么在这步棋后，根据棋盘状态使用位置评估器，或在下棋后运行一个更深层次的 Monte Carlo 模拟（称为“rollout”）去阅读这一步棋后将来的情况，它使用快速——阅读移动选择器，加快搜索。这两种方法会产生对下棋质量的独立的猜测。 AlphaGo使用单个参数，——一个“混合系数”，来衡量对比这些猜测。全功率AlphaGo使用1：1比例混合两种猜测，用位置评估器和模拟展出来判断下棋质量。"""
    """$ \qquad $ 该论文包含一个简洁的图表，显示了当启动或停止插入以上的大脑和模拟器时，AlphaGo实力如何变化。只使用一个大脑，AlphaGo大概和最好的电脑围棋AI实力相当，但结合所有配置，它可能可以达到人类职业棋手的水平。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片8.webp', caption='图8 MCTS“插件”打开或关闭时，AlphaGo的实力如何变化的。图为 Silver提供。')
    """$ \qquad $  论文详细介绍了团队在通过计算机网络分布计算来加速MCT的过程中做的一些有趣的工程，但这并不会改变基本的算法。文中还列有每一步的数学推理的附录。该算法的某些部分是精确的（无限计算能力的渐近极限），而有些则只是近似值。在实践中，AlphaGo接入更大的计算力量，肯定会变得更加强大，但每个额外的计算单元的提升速度会随着它的强大而变缓。"""

with tab3:
    """- **神经网络AlphaFold的“颠覆性”数据库预测出了智人和20种模式生物的逾35万个结构。**"""
    """$ \qquad $  人类基因组携带了逾2万个蛋白质的指令，但只有约1/3蛋白质的三维结构通过实验方法得到了解析，很多时候，这些蛋白质的结构只确定了其中一部分。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片9.webp', caption='图9 人类中介体复合物一直是结构生物学家难以理解的一种多蛋白系统。来源：Yuan He')
    """$ \qquad $  现在，一种名为AlphaFold的人工智能（AI）工具改变了现状。这款工具由位于伦敦的谷歌姐妹公司DeepMind开发，其预测的结构几乎覆盖了完整的人类蛋白质组（蛋白质组是一个生物表达的全部蛋白质）。除此之外，AlphaFold还预测了许多其他生物的几乎整个蛋白质组——从小鼠到玉米再到疟原虫（见“折叠选项”）。"""
    """$ \qquad $  这次预测的逾35万个蛋白质结构保存在一个公用数据库中，规模将在年底扩大到1.3亿个。虽然这些预测的准确度有高有低，但研究人员认为这些数据或为生命科学领域带来翻天覆地的变化。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片10.webp', caption='图10 来源：EMBL–EBI和https://swissmodel.expasy.org/repository')
    """**问题概述**:"""
    """$ \qquad $  AlphaFold解决的问题是蛋白质折叠问题，可以抽象成如下："""
    """$ \qquad $  输入：Alpha Fold的输入是一个氨基酸序列，每一个位置的元素代表了链上的一个氨基酸单元（一共可以有21种氨基酸单元）"""
    """$ \qquad $  输出：在接收到这个单一序列的输入之后，AlphaFold需要使用算法，预测这一个氨基酸链条会如何折叠，所以输出的是一个拓扑结构。"""
    """$ \qquad $  输出的数据是每一个氨基酸单元和其下一个氨基酸单元在空间中的夹角，由于是三维空间，所以说明一个方位需要2维数据，(φ, ψ)，所以AlphaFold的模型的输出就是一组数量和输入对应的夹角对："""
    st.latex(r"""(\varphi_1,\psi_1),(\varphi_2,\psi_2),(\varphi_3,\psi_3),(\varphi_4,\psi_4),...""")
    """$ \qquad $  如果输入的是k个氨基酸，那么输出的就是k-1个夹角对，比如上文中输入是59个氨基酸组成的氨基酸链，那么就应该输出58个夹角对。"""
    """- **大体算法框架**："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片11.webp', caption='图11 AlphaFold算法框架')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片12.webp', caption='图12 黄色代表特征抽取，绿色代表结构预测，红色代表potential construction，蓝色代表结构生成')

with tab4:
    """$ \qquad $  1.研究背景 蛋白质对生命至关重要，了解他们的结构可以促进对齐功能的理解。"""
    """$ \qquad $  2.传统的研究方法及其问题 实验的方法至今才能解出10万的蛋白质结构，但是氨基酸序列有10亿的序列，但是传统的实验方法解出一个蛋白的结构可能需要花费数月甚至数年的时间。"""
    """$ \qquad $  3.计算方法解决结构问题方法及其问题 精确的计算方法需要去解决这个差距，并使得大规模结构生物学研究成为可能。仅仅根据氨基酸序列预测三维结构是“Protein folding problem"的结构预测部分，已经被研究了50多年。现有方法虽然有一些进步，但是现有的方法远没有达到原子精度，特别是没有同源模板的时候。"""
    """$ \qquad $  4.Alphafold2 论文首先提出了基于计算方法可以以原子精度来预测蛋白质结构甚至是在没有同源模板的情形。Alphafold2在CASP14上显示了非常高的准确性，Alphafold2将有关蛋白质结构的物理和生物知识，利用多序列比对，融入深度学习算法的设计中。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片13.webp', caption='图13 ')
