import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tab1, tab2, tab3, tab4 = st.tabs(['1.ELMO', '2.Bert', '3.GPT', '4.应用'])

with tab1:
    """- 2018年3月提出，解决多义性"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片38.png', caption='图38')
    """- ELMo模型对语句的正向和逆向分别使用了LSTM(双向的LSTM),可以更好的捕获语句中上下文之间的关系，并使用了多层的LSTM结构（上图中是两层），底层的LSTM可以用于捕获句法信息，顶层的LSTM可以用于捕获语义信息。"""
    """- 优势："""
    r"""$ \qquad $ 1.多义词方面的极大改善。"""
    r"""$ \qquad $ 2.效果提升：6个NLP任务中性能都有幅度不同的提升，最高的提升达到25%左右，而且这6个任务的覆盖范围比较广，包含句子语义关系判断，分类任务，阅读理解等多个领域，这说明其适用范围是非常广的，普适性强，这是一个非常好的优点。"""
    """- 劣势："""
    r"""$ \qquad $ 1.在特征抽取器方面，ELMo使用了LSTM而不是新贵Transformer;"""
    r"""$ \qquad $ 2.训练时间长，这也是RNN的本质导致的，和上面特征提取缺点差不多。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片39.png', caption='图39')
with tab2:
    """- Google Brain实习生尤洋把预训练BERT的速度足足提高了64倍。训练时间从4860分钟，变成了76分钟11秒。训练完成后，在机器问答数据集SQuAD-v1上测试一下，F1得分比原来的三天三夜版还要高一点点。"""
    """- 从本质上分析，BERT语言模型就是Transformer模型的编码器部分。"""
    """- BERT的全称是Bidirectional Encoder Representation from Transformers,即双向Transformer的Encoder.模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。"""
    """- BERT模型有3亿个参数。"""
    """- 对比OpenAI GPT(Generative pre-trained transformer)，BERT是双向的Transformer block连接；就像单向RNN和双向RNN的区别，直觉上来讲效果会好一些。"""
    """- 对比ELMo，虽然都是“双向”，但目标函数其实是不同的。ELMo是分别以[公式] 和 [公式] 作为目标函数，独立训练处两个representation然后拼接，而BERT则是以 [公式] 作为目标函数训练LM。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片40.png', caption='图40')

with tab3:
    """- GPT的核心思想是先通过无标签的文本去训练生成语言模型，再根据具体的NLP任务（如文本蕴涵、QA、文本分类等），来通过有标签的数据对模型进行fine-tuning。"""
    """- GPT-2,有15亿个参数。"""
    """- 2020年6月，GPT-3，有1750亿个参数。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片41.png', caption='图41')
    """- GPT 使用 Transformer 的 Decoder 结构，并对 Transformer Decoder 进行了一些改动，原本的 Decoder 包含了两个 Multi-Head Attention 结构，GPT 只保留了 Mask Multi-Head Attention。"""
    _, col1, _ = st.columns([1, 1, 1])
    with col1:
        st.image('./pages/图片/图片42.png', caption='图42')
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./pages/图片/图片43.png', caption='图43')
    """- GPT-2 主要就是在 GPT 的基础上，又添加了多个任务，扩增了数据集和模型参数，又训练了一番。我们一般的 NLP 任务，文本分类模型就只能分类，分词模型就只能分词，机器翻译也就只能完成翻译这一件事，非常不灵活。
    这个过程也和人脑很像，人脑是非常稳定和泛化的，既可以读诗歌，也可以学数学，还可以学外语，看新闻，听音乐等等，简而言之，就是一脑多用。"""
    _, col1, _ = st.columns([1, 1, 1])
    with col1:
        st.image('./pages/图片/图片97.png', caption='图97 GPT-2学习效果图')

    """- GPT-3 的模型所采用的数据量之大，高达上万亿，模型参数量也十分巨大，学习之复杂。如图所示："""
    _, col1, _ = st.columns([1, 1, 1])
    with col1:
        st.image('./pages/图片/图片98.png', caption='图98')
    """- GPT-3 里的大模型计算量是 Bert-base 的上千倍。如此巨大的模型造就了 GPT-3 在许多十分困难的 NLP 任务，诸如撰写人类难以判别的文章，甚至编写SQL查询语句，React或者JavaScript代码上优异的表现。GPT-n 系列模型都是采用 decoder 进行训练的，也就是更加适合文本生成的形式。也就是，输入一句话，输出也是一句话。也就是对话模式。"""

    """- $\\textbf{ChatGPT}$"""
    """- 几年前，alpha GO 击败了柯洁，几乎可以说明，强化学习如果在适合的条件下，完全可以打败人类，逼近完美的极限。"""
    """- 强化学习非常像生物进化，模型在给定的环境中，不断地根据环境的惩罚和奖励（reward），拟合到一个最适应环境的状态。"""
    _, col1, _ = st.columns([1, 1, 1])
    with col1:
        st.image('./pages/图片/图片99.png', caption='图99')
    """- NLP 所依赖的环境，是整个现实世界，整个世界的复杂度，远远不是一个19乘19的棋盘可以比拟的。无法设计反馈惩罚和奖励函数，即 reward 函数。除非人们一点点地人工反馈。"""
    """- open-ai 的 chatGPT 就把这事给干了。撒钱，找40个外包，标起来！"""
    """- 如何构建一个 reward 函数，具体就是让那40名外包人员不断地从模型的输出结果中筛选，哪些是好的，哪些是低质量的，这样就可以训练得到一个 reward 模型。通过reward 模型来评价模型的输出结果好坏。"""
    """- 只要把预训练模型接一根管子在 reward 模型上，预训练模型就会开始像感知真实世界那样，感知reward。"""

with tab4:
    """- 机器翻译："""
    """$ \qquad $ 对于机器来说，翻译就是一个解码后再编码的过程。如果把英语翻译成中文，就要先把英语原文解码成“神经代码”，再编码生成中文。"""
    """$ \qquad $ 语言模型的变体：条件+语言模型=翻译模型"""
    """$ \qquad $ 将整个源文本进行符号化处理，并以一个固定的特殊标记作为翻译模型的开始符号。之后同样地，对这两个序列进行联合建模，得到概率最大的下一个译词。"""
    """$ \qquad $ 同样地，将生成的词加入译文序列，然后重复上述步骤反复迭代，不断生成之后的每一个译词。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片44.gif', caption='图44')
    """$ \qquad $ 当代表句子终止的符号被模型选择出来之后，停止迭代过程，并进行反符号化处理，得到自然语句译文。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片45.gif', caption='图45')

    """- 语音识别："""
    """$ \qquad $ 语音识别的目的是将人类语音中的词汇内容转换为计算机可读的输入内容。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片46.png', caption='图46')
