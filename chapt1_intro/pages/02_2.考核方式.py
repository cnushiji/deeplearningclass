import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


st.set_page_config(layout='wide')

"""- 考核方式："""
r"""
$ \quad $最终成绩由3部分构成，

$ \qquad $1.平时作业20分

$ \qquad $2.课程大作业40分

$ \qquad $3.期末小组汇报40分，无笔试期末考试
"""


