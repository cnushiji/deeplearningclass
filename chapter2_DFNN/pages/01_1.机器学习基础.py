import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

tab1, tab2, tab3 = st.tabs(['1.场景和任务', '2.学习目标', '3.数据模型和性能评价'])

with tab1:
    """- 机器学习简介："""

    r"""$ \qquad $ 机器学习算法是一种能够从数据中学习的算法。"""
    r"""$ \qquad $ 对于某类任务 T 和性能度量P，一个计算机程序被认为可以从经验 E 中学习是指，通过经验 E 改进后，它在任务 T 上由性能度量 P 衡量的性能有所提升。
 $ \qquad $ —Mitchell，1997 """


    """- 机器学习三要素："""
    r"""$ \qquad $ 经验E：数据集(程序与自己下几万次跳棋) ，由大量样本构成。"""
    r"""$ \qquad $ 任务T：通俗的说就是我们希望达成的目标。比如机器翻译：输入一种语言序列，给出另一种语言序列。
"""
    r"""$ \qquad $ 性能度量P：为了评估机器学习算法的性能而设计的定量度量。如与新对手玩跳棋时赢的概率。"""

    """- 场景——社交媒体"""
    r"""$ \qquad $ 首先，借助机器学习，数十亿用户可以有效地参与社交媒体网络。机器学习在推动社交媒体平台从个性化新闻提要到投放特定于用户的方面发挥着关键作用。"""
    r"""$ \qquad $ 例如，Facebook 的自动标记功能采用图像识别来识别您朋友的脸并自动标记他们。该社交网络使用 ANN 识别用户联系人列表中熟悉的面孔，并促进自动标记。"""
    r"""$ \qquad $ 同样，LinkedIn 知道您应该何时申请下一个职位，您需要联系谁，以及您的技能与同行相比排名如何。所有这些功能都是通过机器学习实现的。"""

    """- 任务——分类"""
    r"""$ \qquad $ 在机器学习任务的解决中，分类就是对将要预测的事情进行建模的过程，针对给定输入数据预测了类别标签。"""
    r"""$ \qquad $ 常见的分类问题有："""
    r"""$ \qquad \qquad $ 1.判断电子邮件是否为垃圾邮件。"""
    r"""$ \qquad \qquad $ 2.给定一个手写字符，判断是否是已知字符。"""
    r"""$ \qquad \qquad $ 3.根据网站最近的用户行为特征，判断是否流失。"""
    r"""$ \qquad $ 分类需要先找到数据样本点中的分界线，再根据分界线对新数据进行分类，分类数据是离散的值，比如图片识别、情感分析等领域会经常用到分类任务。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片5.jpg', caption='图5')

    """- 二分类任务："""
    r"""$ \qquad $ 二分类任务是指具有两个类别标签分类任务。"""
    r"""$ \qquad $ 例如：垃圾邮件检测（是否为垃圾邮件），客户流失预测（是否流失），客户购买欲预测（购买或不购买）。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片1.png', caption='图1')
    """- 多分类任务："""
    r"""$ \qquad $ 多类别分类是指具有两个以上类别标签的分类任务。"""
    r"""$ \qquad $ 例如：人脸识别分类，植物种类分类，手写数字识别分类"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片2.png', caption='图2')
    """- 多标签分类任务："""
    r"""$ \qquad $ 多标签分类是指具有两个或多个分类标签的分类任务，其中每个示例可以预测一个或多个分类标签。"""
    r"""$ \qquad $ 例如：每个输入样本都有两个输入特征。一共有三个类别，每个类别可能带有两个标签（0或1）之一。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片3.png', caption='图3')
    """- 样本不平衡分类任务："""
    r"""$ \qquad $ 样本不平衡分类是指分类任务，其中每个类别中的示例数不均匀分布。通常，不平衡分类任务是二分类任务，其中训练数据集中的大多数示例属于正常类，而少数示例属于异常类。"""
    r"""$ \qquad $ 例如：欺诈识别，离群值检测，肺炎识别"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片4.png', caption='图4')

    """- 任务——回归"""
    r"""$ \qquad $ 回归是对已有的数据样本点进行拟合，再根据拟合出来的函数，对未来进行预测。回归数据是连续的值，比如商品价格走势的预测就是回归任务。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片6.png', caption='图6')
    """- 分类任务与回归任务的区别——需要预测的值的类型"""
    r"""$ \qquad $ 回归任务，是对 连续值 进行预测（比如 多少）"""
    r"""$ \qquad $ 分类任务，是对 离散值 进行预测（比如 是不是，属不属于，或者 属于哪一类）"""
    r"""$ \qquad $ 比如："""
    r"""$ \qquad \qquad $ 预测 明天的气温是多少度，这是一个回归任务"""
    r"""$ \qquad \qquad $ 预测 明天会不会下雨，就是一个分类任务"""

with tab2:
    """- 为什么需要目标函数？"""
    r"""$ \qquad $ 几乎所有的机器学习算法最后都归结为求解最优化问题，以达到我们想让算法达到的目标。"""
    r"""$ \qquad $ 为了完成某一目标，需要构造出一个目标函数来，然后让该函数取极大值或极小值（也就是优化），从而得到机器学习算法的模型参数。"""
    r"""$ \qquad $ 如何构造出一个合理的目标函数，是建立机器学习算法的关键，一旦目标函数确定，接下来就是求解最优化问题，这在数学上一般有现成的方案。"""
    st.latex(r"""y=f(x；\theta)""")
    r"""$ \qquad $ 其中$ \theta $是模型的参数，如何确定它的值，是学习算法的核心。"""

with tab3:
    """- 数据集："""
    r"""$ \qquad $ 训练集：用来训练模型的样本的集合"""
    r"""$ \qquad $ 验证集：用来确定网络结构或者控制模型复杂程度的参数 """
    r"""$ \qquad $ 测试集：评估模型在未观测数据上的性能 """
    r"""$ \qquad $ 独立同分布假设：训练集和测试集中的样本是独立同分布的，这个分布称为数据生成分布。"""

    """- 学习方式："""
    r"""$ \qquad $ 监督学习：训练集中每一个样本都有一个对应的标签(label)"""
    r"""$ \qquad $ 半监督学习：训练集中只有部分样本有标签"""
    r"""$ \qquad $ 无监督学习：所有训练样本都没有标签"""

    """- 模型容量："""
    r"""$ \qquad $ 泛化：在先前未观测到的输入上表现良好的能力"""
    r"""$ \qquad $ 训练误差：假设学习到的模型是$ y=\hat{f}(x) $，训练误差是模型$ y=\hat{f}(x) $关于训练数据集的平均损失：
"""
    st.latex(r"""R_{emp}{(\hat{f})}=\frac{1}{N}\sum_{i=1}^N{L(y_i,\hat{f}{(x)})}""")
    r"""$ \qquad \qquad $ 其中N是训练样本容量。"""
    r"""$ \qquad $ 泛化误差：模型$ y=\hat{f}(x) $关于测试数据集的平均损失，即"""
    st.latex(r"""e_{test}=\frac{1}{N'}\sum_{i=1}^{N'}{L(y_i,\hat{f}{(x)})}""")
    r"""$ \qquad \qquad $ 其中$ 𝑁^‘$是测试样本容量。"""
    r"""$ \qquad $ 模型容量：通俗的说是指模型拟合各种函数的能力"""

    """- 分类问题性能评价："""
    """$ \qquad $ 1.混淆矩阵： """
    r"""$ \qquad \qquad $ 假设是二分类问题，分为正例(positive)和负例(negative)。"""
    r"""$ \qquad \qquad $ Ture positive(TP)：真正，实际为正例且被分类器划分为正例的实例数（划分正确）。"""
    r"""$ \qquad \qquad $ False positive(FP)：假正，实际为负例但被分类器划分为正例的实例数（划分错误）。"""
    r"""$ \qquad \qquad $ False negatives(FN)：假负，实际为正例但被分类器划分为负例的实例数（划分错误）。"""
    r"""$ \qquad \qquad $ True negatives(TN)：真负，实际为负例但被分类器划分为负例的实例数（划分正确）。"""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片7.png', caption='图7')

    """$ \qquad $ 2.准确率： """
    r"""$ \qquad \qquad $ 准确率是分类器分类正确的样本数量与样本总数的比。"""
    st.latex(r"""Accuracy = \frac{TP+TN}{TP+TN+FP+FN}""")
    """$ \qquad $ 3.错误率："""
    r"""$ \qquad \qquad $ 错误率则正好与准确率含义相反。"""
    st.latex(r"""Accuracy = \frac{FP+FN}{TP+TN+FP+FN}""")
    """$ \qquad $ 4.精准率： """
    r"""$ \qquad \qquad $ 精准率是指被分类器判定为正类样本中真正的正类样本所占的比重。"""
    st.latex(r"""Precision = \frac{TP}{TP+FP}""")
    """$ \qquad $ 5.召回率： """
    r"""$ \qquad \qquad $ 召回率是指被分类器正确判定的正类样本占总的正类样本的比重。"""
    st.latex(r"""Recall = \frac{TP}{TP+FN}""")
    r"""$ \qquad $6.F-score $ \qquad $ 7.平均 $ \qquad $ 8.P-R曲线 $ \qquad $ 9.ROC曲线 $ \qquad $ 10.AUC(area under curve)"""

    """- 回归问题性能评价："""
    r"""$ \qquad $ 1.平均绝对误差(MAE)："""
    r"""$ \qquad \qquad $ 计算每一个样本的预测值和真实值的差的绝对值，然后求和再取平均值。这个指标是对绝对误差损失的预期值。"""
    r"""$ \qquad \qquad $ MAE对极端值比较敏感，即MAE 对异常值更加稳健，因为它不使用平方。"""
    st.latex(r"""MAE = \frac{1}{n}\sum_{i=1}^n{|y_i-\hat{y}_i|}""")
    r"""$ \qquad $ 2.均方误差(MSE)："""
    r"""$ \qquad \qquad $ （Mean Squared Error）计算每一个样本的预测值与真实值差的平方，然后求和再取平均值。该指标对应于平方(二次)误差的期望。是线性回归的损失函数，在线性回归的时候我们的目的就是让这个损失函数最小。"""
    r"""$ \qquad \qquad $ 受到异常值的影响很大。"""
    st.latex(r"""MSE = \frac{1}{n}\sum_{i=1}^n{(y_i-\hat{y}_i)^2}""")
    r"""$ \qquad $ 3.均方根误差(RMSE)："""
    r"""$ \qquad \qquad $ 它测量的是预测过程中预测错误的标准标准偏差（标准偏差是方差的算术平方根，而方差是离均平方差的平均数）。"""
    st.latex(r"""RMSE(X,h)=\sqrt{\frac{1}{m}\sum_{i=1}^m{(h(X^{(i)})-y^{(i)})^2}}""")
    r"""$ \qquad \qquad $ 其中，M是RMSE数据集中实例的个数；$ X^{(i)} $是数据集第i个实例的所有特征值的向量，$ y^{(i)} $是它的标签；
H是系统的预测函数，也称为假设。当系统受到一个实例的特征向量$ X^{(i)} $，就会输出这个实例的一个预测值$ \tilde{y}=h(X^{(i)}) $。"""
    r"""$ \qquad $ 4.平均绝对百分比误差(MAPE)："""
    r"""$ \qquad \qquad $ （Mean Absolute Percentage Error）这个指标是对相对误差损失的预期值。所谓相对误差，就是绝对误差和真值的百分比。"""
    st.latex(r"""MAPE = \frac{1}{n}\sum_{i=1}^n{\frac{|y_i-\hat{y}_i|}{|y_i|}}""")
    r"""$ \qquad $ 5.均方误差对数(MSLE)："""
    r"""$ \qquad \qquad $ （Mean Squared Log Error）该指标对应平方对数(二次)差的预期。"""
    r"""$ \qquad \qquad $ 当数据当中有少量的值和真实值差值较大的时候，使用log函数能够减少这些值对于整体误差的影响。"""
    st.latex(r"""MSLE = \frac{1}{n}\sum_{i=1}^n{(log(1+y_i)-log(1+\hat{y}_i))^2}""")
    r"""$ \qquad $ 6.中位绝对误差(MedAE)："""
    r"""$ \qquad \qquad $ （Median Absolute Error）通过取目标和预测之间的所有绝对差值的中值来计算损失。"""
    st.latex(r"""MedAE = median(|y_1-\hat{y}_1|,...,|y_n-\hat{y}_n|)""")
    r"""$ \qquad $ 7.R Squared："""
    r"""$ \qquad \qquad $ 又叫可决系数(coefficient of determination)/拟合优度，取值范围为0~1，反映的是自变量 x 对因变量 y 的变动的解释的程度。"""
    r"""$ \qquad \qquad $ 越接近于1，说明模型拟合得越好。"""
    st.latex(r"""R^2=1-\frac{MSE(y,\hat{y})}{Var(y)}=\frac{\sum_{i=1}^n{(y_i-\hat{y}_i)^2}}{\sum_{i=1}^n{(y_i-\bar{y})^2}}=\frac{SSE}{SST}=1-\frac{SSR}{SST}""")
    r"""$ \qquad \qquad $ 其中，"""
    r"""$ \qquad \qquad \qquad  SST=\sum_{i=1}^{n}{(y_{i}-\bar{y})^2} $ 表示的是y的变动的程度，正比于方差;"""
    r"""$ \qquad \qquad \qquad  SSR=\sum_{i=1}^{n}{(y_{i}-\hat{y_{i}})^2} $ 表示的是模型和真实值的残差； """
    r"""$ \qquad \qquad \qquad  SSE=\sum_{i=1}^{n}{(\hat{y}_{i}-\bar{y})^2} $ 表示的是模型对y的变动的预测。"""
    r"""$ \qquad \qquad \qquad $ SST=SSR+SSE. """
    """- 欠拟合："""
    r"""$ \qquad $ 度量泛化能力的好坏，最直观的特征就是模型的过拟合和欠拟合。"""
    r"""$ \qquad $ 欠拟合：模型不能够在训练集上获得足够低的误差，不能够很好的拟合训练数据。"""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片8.png', caption='图8')
    """- 过拟合："""
    r"""$ \qquad $ 训练误差和测试误差的差距太大，泛化性能不好。"""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./chapter2_DFNN/pages/图片/图片9.png', caption='图9')

    """- 如何解决欠拟合？"""
    r"""$ \qquad $ 不断训练：欠拟合基本都会发生在训练刚开始的时候。"""
    r"""$ \qquad $ 增加网络复杂度：网络复杂度过低，会导致欠拟合。"""
    r"""$ \qquad $ 在模型中增加特征等。"""

    """- 为什么会出现过拟合现象？"""
    r"""$ \qquad $ 训练数据集样本单一，样本不足。"""
    r"""$ \qquad $ 训练数据中噪声干扰过大。"""
    r"""$ \qquad $ 模型过于复杂，即模型的表达能力太强了。"""

    """- 如何防止过拟合？"""
    r"""$ \qquad $ 显著减少测试误差而不过度增加训练误差。"""
    r"""$ \qquad $ 获取更多的训练数据。"""
    r"""$ \qquad $ 使用正则化方法。"""

    """- 如何实现机器学习目标——估计"""
    r"""$ \qquad $ 统计领域为我们提供了很多工具来实现机器学习目标，不仅可以解决训练集上的任务，还可以泛化。基本的概念，例如参数估计、偏差和方差，对于正式地刻画泛化、欠拟合和过拟合都非常有帮助。"""
    r"""$ \qquad $ 1.点估计："""
    r"""$ \qquad \qquad $ 令 $ \left\{𝑥_{(1)},…,𝑥_{(m)} \right\} $是m个独立同分布（i.i.d）的数据点。 """
    r"""$ \qquad \qquad $ 点估计（point estimator）或统计量（statistics）是这些数据的任意函数："""
    st.latex(r"""\hat{\theta}=g(x^{(1)},...,x^{(m)})""")
    r"""$ \qquad \qquad $ 一个良好的估计量的输出会接近生成训练数据的真实参数$\theta$。"""
    r"""$ \qquad $ 2.函数估计："""
    r"""$ \qquad \qquad $ 点估计也可以指输入和目标变量之间函数关系的估计，将这种类型的点估计称为函数估计。"""
    r"""$ \qquad \qquad $ 我们试图从输入向量x预测变量y，假设有一个函数$f(x)$表示y和x之间的近似关系。在函数估计中，我们感兴趣的是用模型估计去近似f，或者估计$\hat{f}$。函数估计和估计参数$\theta$是一样的，函数估计$\hat{f}$是函数空间中的一个点估计。"""
    r"""$ \qquad \qquad $ 例如：输入数据$ \left\{(𝑥_1,y_1),…,(𝑥_n,y_n)\right\} $近似一个函数$ 𝑓=𝑎𝑥+𝑏𝑥^2+𝑐𝑥^3 $，用模型估计去近似𝑓，或者得到$\hat{f}$。"""

    r"""$ \qquad $ 3.最大似然估计："""
    r"""$ \qquad \qquad $ 考虑一组含有m个样本的数据集$ X=\left\{𝑥^{(1)},…,𝑥^{(m)}\right\} $，独立地由未知的真实数据生成分布$ 𝑃_{𝑑𝑎𝑡𝑎}{(𝑥)} $生成。"""
    r"""$ \qquad \qquad $ 令$ 𝑃_{𝑑𝑎𝑡𝑎}{(𝑥;\theta)} $ 是一簇由$ \theta $ 确定在相同空间上的概率分布。"""
    r"""$ \qquad \qquad $ 换言之，$ 𝑃_{𝑑𝑎𝑡𝑎}{(𝑥;\theta)} $将任意输入x映射到实数来估计真实概率$ 𝑃_{𝑑𝑎𝑡𝑎}{(𝑥)} $。"""
    r"""$ \qquad \qquad $ 对$ \theta $的最大似然估计被定义为："""
    st.latex(r"""\theta_{𝑀𝐿}=\mathop{argmax}\limits_{\theta}{\pi_{𝑖=1}^𝑚{𝑃_{𝑚𝑜𝑑𝑒𝑙}{(𝑥^{(𝑖)};\theta)}}}""")
    r"""$ \qquad \qquad $ 为了避免计算中出现数值下溢，将乘积转化成了便于计算的求和形式："""
    st.latex(r"""\theta_{𝑀𝐿}=\mathop{argmax}\limits_{\theta}\sum_{i=1}^m{log{P_{model}(x^{(i)};\theta)}}""")
    r"""$ \qquad \qquad $ 我们可以除以m得到和训练数据经验分布$ \hat{p}_{data} $相关的期望作为准则："""
    st.latex(r"""\theta_{ML}=\mathop{argmax}\limits_{\theta}{E_{x\sim{\hat{P}_{data}}}}{log{P_{model}(x;\theta)}}""")
    r"""$ \qquad \qquad $ 性质：一致性，渐进无偏性，近似效率高。"""
    r"""$ \qquad $ 4.条件对数似然："""
    r"""$ \qquad \qquad $ 最大似然估计很容易地扩展到估计条件概率$ 𝑃(𝑦|𝑥;\theta) $,从而给定x预测y。实际上这是最常见的情况，因为这构成了大多数监督学习的基础。"""
    r"""$ \qquad \qquad $ 如果X表示所有的输入，Y表示我们观测到的目标，那么条件最大似然估计是："""
    st.latex(r"""\theta_{ML}=\mathop{argmax}\limits_{\theta}{P(Y|X;\theta)}""")
    r"""$ \qquad \qquad $ 如果假设样本是独立同分布的，那么这可以分解成："""
    st.latex(r"""\theta_{ML}=\mathop{argmax}\limits_{\theta}\sum_{i=1}^m{log{P(y^{(i)}|x^{(i)};\theta)}}""")
    r"""$ \qquad $ 5.Bayes估计："""
    r"""$ \qquad \qquad $ 贝叶斯用概率反映知识状态的确定性程度"""
    r"""$ \qquad \qquad $ 假设我们有一组数据样本 $ \left\{𝑥_{(1)},…,x_{(m)}\right\} $。通过贝叶斯规则结合数据似然$ P(𝑥_{(1)},…,x_{(m)}|\theta) $和先验，我们可以恢复数据对我们关于$\theta$信念的影响："""
    st.latex(r"""P(\theta|x_{(1)},...,x_{(m)})=\frac{P(𝑥_{(1)},…,x_{(m)}|\theta)P(\theta)}{P(x_{(1)},...,x_{(m)})}""")
    r"""$ \qquad \qquad $ 在贝叶斯估计常用的情景下，先验开始是相对均匀的分布或高熵的高斯分布，观测数据通常会使后验的熵下降，并集中在参数的几个可能性很高的值。"""
    r"""$ \qquad $ Bayes估计与最大似然估计的重要区别："""
    r"""$ \qquad \qquad $ 不像最大似然方法预测时使用$\theta$的点估计，贝叶斯方法使用$\theta$的全分布。"""
    r"""$ \qquad \qquad $ 例如，在观测到m个样本后，下一个数据样本$ 𝑥_{𝑚+1} $的预测分布如下："""
    st.latex(r"""p(x^{(m+1)}|x^{(1)},...,x^{(m)})=\int{p(x^{(m+1)}|\theta)p(\theta|x^{(1)},...,x^{(m)})d\theta}""")
    r"""$ \qquad \qquad $ 贝叶斯先验分布：先验能够影响概率质量密度朝参数空间中偏好先验的区域偏移。实践中，先     验通常表现为偏好更简单或更光滑的模型。对贝叶斯方法的批判认为先验是人为主观判断影响预测的来源。当训练数据很有限时，贝叶斯方法通常泛化得更好，但是当训练样本数目很大时，通常会有很大的计算代价。"""

    r"""$ \qquad $ 6.最大后验估计(MAP)："""
    r"""$ \qquad \qquad $ MAP是Bayes估计的一种，对于大多数有意义的模型而言，大多数涉及到贝叶斯后验的计算是非常棘手的，点估计提供了一个可行的近似解。"""
    r"""$ \qquad \qquad $ MAP估计选择后验概率最大的点："""
    st.latex(r"""\theta_{MAP}=\mathop{argmax}\limits_{\theta}p(\theta|x)=\mathop{argmax}\limits_{\theta}{log{p(x|\theta)}}+log(p(\theta))""")
    r"""$ \qquad \qquad \qquad $ 上式中$log{p(x|\theta)}$对应标准的条件对数似然项，$log{p(\theta)}$对应先验分布项。"""


