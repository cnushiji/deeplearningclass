import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

""" 本章<<深度前馈神经网络>>分3次课来讲，主要介绍机器学习基础和MLP"""
""" 机器学习基础介绍了机器学习的场景和任务、学习目标、数据模型和性能评价"""
""" MLP介绍了MLP的网络结构、优化、反向传播、初始化、正则化、示例——mnist"""