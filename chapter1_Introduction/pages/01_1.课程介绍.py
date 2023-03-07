import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

tab1, tab2, tab3 = st.tabs(['1.课程简介', '2.课程内容要点', '3.参考书'])

with tab1:
    """- 课程简介:"""

    r"""$ \qquad $人工智能是21世纪的代表性技术，被认为将引领和推动第四次工业革命，受到各国政府和各行业的高度重视。深度学习是人工智能的重要分支，目前其发展和实际应用方兴未艾。因此了解和掌握深度学习技术的基本原理和应用实践具有重要意义。
"""


with tab2:
    """- 课程内容要点："""
    r"""
    $ \qquad $1.DL基础：本课程将先对深度学习技术的发展历史做一个全面的回顾，然后介绍深度学习的基本概念，网络结构，优化方法，正则化方法以及网络训练技巧等内容。

    $ \qquad $2.代表性模型介绍：在前面学习的基础上我们将介绍在诸多应用场景的代表性深度学习方法、卷积神经网络、循环神经网络、生成式神经网络。

    $ \qquad $3.应用介绍：在学习深度学习理论知识的同时，我们将基于具体应用实例开展深度学习的编程实践。
    """


with tab3:
    """- 参考书："""
    r"""
    $ \qquad $1.《Deep Learning》，Ian Goodfellow, et al.

    $ \qquad $2.《Neural  networks and deep learning》，Michael Nielsen

    $ \qquad $3.《python编程导论》
    """


