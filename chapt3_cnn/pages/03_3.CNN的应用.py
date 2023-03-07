import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(['1.图像分类', '2.检测识别', '3.三维重建', '4.图像分割', '5.图像去噪和超分辨', '6.图像生成'])

with tab1:
    """- 图像分类的定义：基于图像的内容对图像进行标记，通常会有一组固定的标签，而模型要对输入的图像预测一个标签。例如进行minist手写数字分类、cifar10图片分类等。"""
    """- 图像分类的损失函数："""
    r"""$ \qquad $ 1.如何输出概率？softmax"""
    r"""$ \qquad $ 2.类别标签的one-hot编码"""
    r"""$ \qquad $ 3.如何度量分类损失？crossentropy"""
    """- 图像分类的应用————ResNet: """
    r"""$ \qquad $ 深度残差网络是由微软研究院的何恺明、张祥雨、任少卿、孙剑等人提出的，是CNN图像史上的一件里程碑事件。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片25.png', caption='图25 ResNet在ILSVRC和COCO 2015上的战绩')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片26.png', caption='图26 ImageNet分类Top-5误差')
    r"""$ \qquad $ 1.深度网络中的退化问题:"""
    r"""$ \qquad \qquad $ 1. 从经验来看，网络的深度对模型的性能至关重要，当增加网络层数后，网络可以进行更加复杂的特征模式的提取，所以当模型更深时，理论上可以取得更好的结果。"""
    r"""$ \qquad \qquad $ 2. 实验发现深度网络出现了退化问题（Degradation problem）:网络深度增加时，网络准确度出现饱和，甚至出现下降。如图："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片27.png', caption='图27 20层与56层网络在CIFAR-10上的误差')
    """$ \qquad $ 2.残差学习单元： """
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片28.png', caption='图28 残差学习单元')
    """$ \qquad $ 3.不同的残差单元："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片29.png', caption='图29 不同的残差单元')
    """$ \qquad $ 4.ResNet的网络结构："""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./pages/图片/图片30.png', caption='图30 ResNet的网络结构')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片31.png', caption='图31 18-layer和34-layer的网络效果')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片32.png', caption='图32 改进后的残差单元及效果')

with tab2:
    """- 目标检测的定义：从输入图像中提取感兴趣区域，并绘制边界框。通常和图像识别结合在一起，对提取的目标区域进行识别。"""
    """- 图像识别的定义：输入图像中包含多个待识别对象，模型通常要对这些对象进行检测，框出对象，然后给出被框对象的类别。"""
    """- 从计算机视觉的角度看，目标检测是分类+定位，从机器学习的角度看，目标检测是分类+回归。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片33.png', caption='图33')
    r"""- 检测识别的方法："""
    r"""$ \qquad $ 1.传统方法：Hog/SIFT视觉特征提取+svm分类。"""
    r"""$ \qquad $ 2.基于深度学习的目标检测识别方法: one-stage、two-stage。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片34.jpg', caption='图34 Two-stage算法')
    """- 检测识别的应用————1.faster R-CNN："""
    r"""$ \qquad $ R-CNN家族："""
    r"""$ \qquad \qquad $ Ross B.Girshick在2016年提出了新的Faster RCNN, 综合性能有较大提高，在检测速度方面尤为明显。"""
    r"""$ \qquad \qquad $ R-CNN"""
    r"""$ \qquad \qquad $ fast R-CNN"""
    r"""$ \qquad \qquad $ faster R-CNN"""
    r"""$ \qquad \qquad $ mask R-CNN"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片35.jpg', caption='图35')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片36.jpg', caption='图36')
    r"""$ \qquad \qquad $ 设计 Region Proposal Networks（RPN），利用 CNN 卷积操作后的特征图生成候选区，代替了Selective Search、EdgeBoxes 等方法，速度上提升明显。"""
    r"""$ \qquad \qquad $ 训练 Region Proposal Networks 与检测网络（Fast R-CNN）共享卷积层，大幅提高网络的检测速度。"""
    r"""$ \qquad \qquad $ 详细介绍见：https://zhuanlan.zhihu.com/p/31426458"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./pages/图片/图片37.png', caption='图37 Faster RCNN基本结构（4个）')
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片38.png', caption='图38 VGG16模型中的faster_rcnn_test.pt的网络结构')
    """- 检测识别的应用————2.YoLo系列："""
    r"""$ \qquad $ YoLo系列是one-stage算法，即直接从图片生成位置和类别，没有明显的生成候选框的过程。YOLO是You only look once的缩写。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片39.jpg', caption='图39')
    r"""$ \qquad $ YoLo思想：YoLo固定维度的办法是把模型的输出划分成网格形状，每个网格中的cell都可以输出物体的类别和bounding box的坐标，如下图：
"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片40.png', caption='图40')
    r"""$ \qquad $ YoLo核心思想: """
    r"""$ \qquad \qquad $ 物体落在哪个cell,哪个cell就负责预测这个物体。"""
    r"""$ \qquad \qquad $ 训练阶段，如果物体中心络再这个cell,那么就给这个cell打上这个物体的label(包括xywh和类别)。即，在训练阶段，就教会cell要预测图像中的哪个物体。"""
    r"""$ \qquad \qquad $ 测试阶段，因为在训练阶段已经教会了cell去预测中心落在该cell中的物体，那么cell自然也会这么做。"""
    _, col1 = st.columns([1, 2])
    with col1:
        st.image('./pages/图片/图片41.png', caption='图41')
    r"""$ \qquad $ YoLo模型: """
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./pages/图片/图片42.png', caption='图42')
    r"""$ \qquad $ YoLo模型架构:"""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./pages/图片/图片43.png', caption='图43')

with tab3:
    """- 三维重建的定义：用相机拍摄真实世界的物体、场景，并通过计算机视觉技术进行处理，从而得到物体的三维模型。英文术语：3D Reconstruction。"""
    """- 三维重建涉及的主要技术有：多视图立体几何、深度图估计、点云处理、网格重建和优化、纹理贴图、马尔科夫随机场、图割等。
基本上计算机视觉的技术都会有涉及。 2D的分割，分类在3D中也会用到。"""

    """- 三维重建从算法角度看有哪些方向?"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片51.png', caption='图51')
    r"""$ \qquad $ MVS是什么?"""
    r"""$ \qquad \qquad $ Multi-View Stereo，用RGB信息重建三维几何模型。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片52.png', caption='图52')
    r"""$ \qquad \qquad $ 输入：就是一系列的RGB照片，这些照片可能存在一些重合势场。将他们的pose计算出来，然后进行一个三维模型的重建，最后进行纹理贴图（非必要）。"""
    """- 三维重建的流程："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片53.png', caption='图53')
    r"""$ \qquad $ 1.输入是images和pose,其中pose是通过SFM(Structure-fromMotion )/Slam(Simultaneous Localization and Mapping ) 来做的。"""
    r"""$ \qquad $ 2.通过上述两种为输入信息，我们可以计算出深度图。"""
    r"""$ \qquad $ 3.得到深度图后，我们再进行点云融合。"""
    r"""$ \qquad $ 4.得到3D点云图后，我们对他进行一个3D曲面的构建。"""
    r"""$ \qquad $ 5.再进行网格优化。"""
    r"""$ \qquad $ 6.最后进行纹理贴图。"""
    """$ \qquad $ 位姿计算："""
    """$ \qquad \qquad $ 位姿计算可以简单理解为：每帧图像的相对位置。"""
    """$ \qquad \qquad \qquad $ SLAM：可以实时去做；SFM：用于离线（COLMAP）"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片54.png', caption='图54')

    """$ \qquad $ 3D重建："""
    """$ \qquad \qquad $ 根据SFM/SLAM计算的位姿，进行稠密重建恢复场景几何信息。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片55.png', caption='图55')

    """- 为什么要应用基于深度学习的3D重建？"""
    """$ \qquad $ 传统的基于光度一致性的MVS重建："""
    """$ \qquad \qquad $ 优点：精度高、对硬件要求低、测量范围大"""
    """$ \qquad \qquad $ 局限性：无纹理（白墙）、透明/反光（玻璃）、重复纹理（铁栅栏）"""
    """$ \qquad $ 基于深度学习的MVS:"""
    """$ \qquad \qquad $ 通过大量的数据去学习特征，它会参考全局语义的信息，能更好的帮助我们进行重建。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片56.png', caption='图56')
    """$ \qquad \qquad $ 优点：基于学习的特征——匹配更鲁棒、基于shape先验——重建更完整。"""
    """$ \qquad \qquad $ 局限性：内存限制，难以重建高分辨率；依赖大数据。"""

    """- 基本概念介绍："""
    r"""$ \qquad $ 深度图（depth）/视差图(disparity)："""
    r"""$ \qquad \qquad $ 深度图：图像中每个点到相机的距离。"""
    r"""$ \qquad \qquad $ 视差图：同一场景在两个相机下成像的像素的位置偏差。二者的关系："""
    st.latex(r"""depth=\frac{bf}{dis}""")
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片57.png', caption='图57')
    r"""$ \qquad $ 三维点云：三维点云是某个坐标系下的点的数据集包含了丰富的信息，包括三维坐标XYZ,颜色RGB等信息"""
    r"""$ \qquad $ 三维网格（mesh）：由物体的邻接点云构成的多边形组成的，通常由三角形、四边形或其他的简单凸多边形组成。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片58.png', caption='图58')
    r"""$ \qquad $ 纹理贴图模型（texture mesh）：带有颜色信息的三维网格模型，所有的颜色信息储存在一张纹理图上，显示时根据每个网格的纹理坐标和对应的纹理图进行渲染得到高分辨率的彩色模型。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片59.png', caption='图59')

    """- 三维重建的深度学习应用————基于NeRF的三维重建："""
    r"""$ \qquad $ NeRF(ECCV 2020)主要贡献："""
    r"""$ \qquad \qquad $ 1.提出一种将具有复杂几何性质和材料的连续场景表示为5D神经辐射场的方法，并将其参数化为基本的MLP网络。 """
    r"""$ \qquad \qquad $ 2.提出一种基于经典体渲染技术的可微渲染方式，论文用它来优化标准RGB图像的表示。"""
    r"""$ \qquad \qquad $ 3.提出位置编码将每个输入5D坐标映射到高维空间，这使得论文能够成功优化神经辐射场来表示高频场景内容。"""
    r"""$ \qquad  \textbf{前言：}$"""
    r"""$ \qquad \qquad  \textbf{1.5D坐标：}$"""
    r"""$ \qquad \qquad $ 论文提出了一种通过使用稀疏的输入图像集优化低层连续体积场景函数(volumetric scene function)的方法，
    从而达到了合成复杂场景新视图的SOTA。论文的算法使用全连接深度网络表示场景，网络的输入是包括3D位置信息(x,y,z)和视角方向$(\theta,\phi)$的单个连续的5D坐标，其输出是该空间位置的体积密度$\aigma$和与视图相关的RGB颜色，接着使用经典的体绘制技术将输出的颜色和密度投影到图像中。
    优化合成新视图所需要的唯一输入是一组具有已知相机位姿的图像，而5D坐标是通过沿着对应像素的相机光线采样所得到的。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片66.png', caption='图66')

    r"""$ \qquad \qquad \textbf{2.坐标变换：}$"""
    r"""$ \qquad \qquad $ 三种坐标系："""
    r"""$ \qquad \qquad \qquad $ 1.世界坐标系：表示物理上的三维世界"""
    r"""$ \qquad \qquad \qquad $ 2.相机坐标系：表示虚拟的三维相机坐标"""
    r"""$ \qquad \qquad \qquad $ 3.图像坐标系：表示二维的图片坐标"""
    r"""$ \qquad \qquad $ 相机坐标$(X_c,Y_c,Z_c)^T$和世界坐标$(X,Y,Z)^T$之间存在如下转换关系："""
    st.latex("""
    \\begin{equation}
  C_c=
        \left[
        \\begin{array}{c}
           X_c \\newline
           Y_c \\newline
           Z_c \\newline
           1
        \end{array}
        \\right]
        =      
        \left[
        \\begin{array}{cccc}
            r_{11} & r_{12} & r_{13} & t_x \\newline
            r_{21} & r_{22} & r_{23} & t_y \\newline
            r_{31} & r_{32} & r_{33} & t_z \\newline
            0      & 0      & 0      & 1
        \end{array}
        \\right]
        \left[
        \\begin{array}{c}
            X \\newline 
            Y \\newline
            Z \\newline
            1
        \end{array}
        \\right] 
    \end{equation}
    """)
    st.latex("""
        \\begin{equation}
        =
        \left[
        \\begin{array}{cccc}
            Col_1 & Col_2 & Col_3 & C_t \\newline 
            0     & 0     & 0     & 1
        \end{array}
        \\right]
        \left[
        \\begin{array}{c}
            X \\newline 
            Y \\newline
            Z \\newline
            1
        \end{array}
        \\right]
        = M_{w2c}C_w
\end{equation}
    """)
    st.latex("""
    Col_i = [r_{1i},r_{2i},r_{3i}]^T,i=[1,2,3] \qquad C_t=[t_x,t_y,t_z]^T
    """)
    r"""$ \qquad \qquad $ 其中，矩阵$M_{w2c}$是一个仿射变换矩阵，也可叫做相机的外参矩阵;$[Col_1,Col_2,Col_3]$包含旋转信息;$C_t$包含平移信息。"""
    r"""$ \qquad \qquad $ 对于二维图片的坐标$[x,y]^T$和相机坐标系下的坐标$[U,V,W]$存在如下转换关系："""
    st.latex("""
    \\begin{equation}
        \left[
        \\begin{array}{c}
            x \\newline 
            y \\newline
            1
        \end{array}
        \\right]
        =
        \left[
        \\begin{array}{ccc}
            f_x & 0 & c_x \\newline 
            0   & f_y  & c_y \\newline
            0   & 0    & 1
        \end{array}
        \\right]
        \left[
        \\begin{array}{c}
            U \\newline 
            V \\newline
            W 
        \end{array}
        \\right]
        = M_{c2i}C_c
    \end{equation}
    """)
    r"""$ \qquad \qquad $ 其中，矩阵$M_{c2i}$为相机的内参矩阵（透视投影矩阵），包含焦距$(f_x,f_y)$以及图像中心点坐标的坐标$(c_x,c_y)。$"""
    r"""$ \qquad \qquad $ 论文使用数据集的配置文件$*.json$中的transform.matrix为上面所述仿射变换矩阵$M_{w2c}$的逆矩阵$M_{c2w}=M^{-1}_{w2c}；而camera_angle_x$是相机的水平视场(horizontal field of view)，可以用于算焦距。"""
    r"""$ \qquad \qquad  \textbf{3.常见图像质量评估指标：}$"""
    r"""$ \qquad \qquad $ 结构相似性SSIM: 一种衡量两幅图像相似度的指标，其值越大则两张图片越相似（范围为[0,1]）。给定两张图片x和y,其计算公式如下："""
    st.latex("""
    SSIM(x,y)=\\frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}
    """)
    r"""$ \qquad \qquad $ 其中，$\mu_x$为x的均值，$\mu_y$为y的均值，$\sigma_x$为x的方差，$\sigma_y$为y的方差，$\sigma_{xy}$为x和y的协方差，$c_1=(K_1L)^2,c_2=(K_2L)^2,K_1=0.01,K_2=0.03$，L为像素值的动态范围(如255)。"""
    r"""$ \qquad \qquad $ 峰值信噪比PSNR:一般用于衡量最大值信号和背景噪音之间的图像质量参考值，其值越大图像失真越少（单位为dB）。一般来说，PSNR高于40dB说明图像质量几乎与原图一样好；在30-40dB之间通常表示图像质量的失真在可接受范围内；在20-30dB之间说明图像质量比较差；PSNR低于20dB说明图像失真严重。给定一个大小为$m\times{n}$
    的原始图像I和带噪声的图像K，首先需要计算这两张图象的均方误差MSE，接着再通过MSE计算峰值信噪比PSNR，其计算公式如下："""
    st.latex("""
    MSE=\\frac{1}{mn}\sum_{i=0}^{m-1}{\sum_{j=0}^{n-1}{[I(i,j)-K(i,j)]^2}}
    """)
    st.latex("""
    PSNR=10\cdot{log_{10}{()\\frac{MAX_I^2}{MSE}}}
    """)
    r"""$ \qquad \qquad $ 其中，$MAX_I$为图像I可能的最大像素值，$I(i,j)$和$K(i,j)$分别为图像I和K对应位置$(i,j)$的像素值。"""

    r"""$ \qquad \qquad $ 学习感知图像块相似度LPIPS:用于计算参考图像块x和失真图像块$x_0$之间的距离，其值越小则相似度越高。LPIPS先提取特征并在通道维度中进行单元归一化，对于l层，我们将得到的结果记为$\hat{y}^l,\hat{y}_0^l\in{R^{H_l\times{H_l}\times{C_l}}}$。
    接着，再利用向量$w_l\in{R^{C_l}}$缩放激活通道并计算$l_2$距离，最后在空间上求平均值，在信道上求和。其公式如下："""
    st.latex("""
    d(x,x_0)=\sum_l{\\frac{1}{H_lW_l}\sum_{h,w}{\Vert w_l\\bigodot{(\hat{y}_{hw}^l-\hat{y}_{0hw}^l)} \Vert_2^2}}
    """)
    r"""$ \qquad  \textbf{网络结构：}$"""
    r"""$ \qquad \qquad $ 神经辐射场 NeRF 是 Neural Radiance Fields 的缩写，其可以简要概括为用一个 MLP 神经网络去隐式地学习一个静态 3D 场景。为了训练网络，针对一个静态场景，需要提供大量相机参数已知的图片。基于这些图片训练好的神经网络，即可以从任意角度渲染出图片的结果。以下为 NeRF 的总体框架："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片67.png', caption='图67')
    r"""$ \qquad \qquad $ 1.获取采样点的5D坐标(x,d)，该坐标包含3D位置信息x=(x,y,z)和视角方向$d=(\theta,\phi)$。"""
    r"""$ \qquad \qquad $ 2.通过位置编码对3D位置$x\in{R^3}$和视角方向$d\in{R^2}$进行相应处理，从而得到编码后的信息$\gamma{(x)}\in{R^{60}}$和$\gamma{(d)}\in{R^{24}}$。"""
    r"""$ \qquad \qquad $ 3.将$\gamma{(x)}$输入到8层全连接层中，每层通道数为256且都经过ReLU函数激活（黑色实箭头）。此外，论文还遵循DeepSDF架构，将通过一个条约链接将输入$\gamma{(x)}$拼接到第5个全连接层的输出中。"""
    r"""$ \qquad \qquad $ 4.在第8个全连接层后添加一个新的全连接层，该层通道数为256且不经过激活函数（橙色箭头）。其输出为维度256的中间特征$F_t$和体素密度$\sigma$(可近似理解为不透明度，值越小越透明)。对于$\sigma$，论文通过激活函数ReLU来确保其为非负的。"""
    r"""$ \qquad \qquad $ 5.将编码后的视角方向$\gamma{(d)}$和中间特征$F_t$拼接起来，然后送入输出通道数为128的全连接层中，再经过激活函数ReLU处理得到特征$F_c$。"""
    r"""$ \qquad \qquad $ 6.将$F_c$送入全连接层中，再经过sigmoid激活函数（黑色虚箭头）处理得到三维的RGB输出。"""
    r"""$ \qquad \qquad $ 7.通过体渲染技术利用上面输出的颜色和体密度来合成图像。由于渲染函数是可微的，所以可以通过最小化合成图像与真实图像之间的误差（residual）来优化场景表示。"""
    r"""$ \qquad \qquad $ NeRF中输出$\sigma$只与位置有关，而颜色RGB还与视角方向相关。"""
    r"""$ \qquad \qquad \textbf{体渲染：}$ """
    r"""$ \qquad \qquad $ 体密度$\sigma{(x)}$可以解释为射线再x位置终止于无穷小的粒子的微分概率（简单理解为不透明度），而相机射线$r(t)=o+td$的期望颜色C(r)在近界$t_n$和远界$t_f$下为："""
    st.latex("""
    C(r)=\int_{t_n}^{t_f}{T(t)\sigma{(r(t))}c(r(t),d)dt},where T(t)=exp(-\int_{t_n}^t{\sigma{(r(s))}ds})
    """)
    r"""$ \qquad \qquad $ 函数$T(t)$表示射线从$t_n$到t沿射线累计透射率，即射线从$t_n$到t不碰到其他任何粒子(particle)的概率。前面累积的体积密度$\int_{t_n}^t{\sigma{(r(s))}ds}$（非负），则T(t)越小，从而降低（因为遮挡）该位置对颜色的影响：
    当前区域体积密度$\sigma{(r(t))}$越高，对颜色影响越大，但其会降低后面区域对颜色的影响（例如前面区域完全不透明，那么后面区域不管什么颜色，都不会影响该方向物体的颜色）。因此，这里根据T(t)来降低相应区域对颜色的影响。"""
    r"""$ \qquad \qquad $ 通过对所需虚拟相机(virtual camera)的每个像素的光线计算这个积分C(r)来为连续的神经辐射场绘制视图，从而得到颜色值。论文用求积法对这个连续积分进行数值估计，而确定性求积(deterministic quadrature)通常用于绘制离散体素网络。因为MLP只会在一些固定的离散位置集上执行，所以它会有限制表示的分辨率
    (representation's resolution)。相反，论文采用分段抽样方法(starting sampling)，将$[t_n,t_f]$划分为N等分，然后从每一块中均匀随机(uniformly at random)抽取一个样本："""
    st.latex("""
    t_i \sim{\mathcal{U}[t_n+\\frac{i-1}{N}(t_f-t_n),t_n+\\frac{i}{N}(t_f-t_n)]}
    """)
    r"""$ \qquad \qquad $ 虽然论文使用离散的样本来估计积分，但因为分段抽样会导致MLP在优化过程中的连续位置被评估，从而能够表示一个连续的场景。论文使用Max在体绘制综述中讨论为正交规则(quadrature rule)和这些离散的样本来估计C(r):"""
    st.latex("""
    \hat{C}(r)=\sum_{i=1}^N{T_i(a-exp(-\sigma_i\delta_i))c_i}, where T_i=exp(-\sum_{j=1}^{i-1}{\sigma_j\delta_j})
    """)
    r"""$ \qquad \qquad $ 其中，$\delta_i=t_{i+1}-t_i$为相邻两个采样点之间的距离。这个从$(c_i,\sigma_i)$值集合计算$\hat{C}(r)$的函数是可微的。因此，基于这样的渲染方式就可以用NeRF函数从任意角度中渲染出图片。"""

    r"""$ \qquad \qquad \textbf{位置编码：}$ """
    r"""$ \qquad \qquad $ 尽管神经网络是通用的函数近似器(universal function approximators)，但其在表示颜色和几何形状方面的高频变化方面表现不佳，这表明深度网络偏向于学习低频函数。论文表明，在将输入传递给网络之前，使用高频函数将输入映射到更高维度的空间，可以更好地你和包含高频变化的数据。为了能有效地提升清晰度，论文引入了
    位置编码，将位置信息映射到高频空间，从而将MLP表示为$F_{\Theta}=F_{\Theta}^{'}\circ{\gamma}$，位置编码$\gamma$表示如下："""
    st.latex("""
    \gamma{(p)}=(sin(2^0\pi{p}),cos(2^0\pi{p}),...,sin(2^{L-1}\pi{p}),cos(2^{L-1}\pi{p}))
    """)
    r"""$ \qquad \qquad $ 其中，$F_{\Theta}^{'}$仍然是一个MLP，而$\gamma$是从R维空间到高维空间$R^{2L}$的映射，$\gamma{(\cdot)}$分别应用于x中的三个坐标值（被归一化到[-1,1]）和笛卡尔视角方向单位向量(Cartesian viewing direction unit vector)d的三个分量（其范围为[-1,1]）。计算$\gamma{(x)}$时
    L=10，即维度为$3\times{2}\times{10}=60$;计算$\gamma{(d)}$时L=4，即维度为$3\times{2}\times{4}=24$。"""
    r"""$ \qquad \qquad \textbf{多层体素级采样：}$ """
    r"""$ \qquad \qquad $ NeRF的渲染过程计算量很大，每条射线都要采样很多点。但实际上，一条射线上的大部分区域都是空区域，或者是被遮挡的区域，对最终的颜色没啥贡献，而原始方法会对这些区域重复取样，从而降低了效率。因此，论文引入多层级体素采样（ Hierarchical volume sampling），采用了一种 “coarse to fine" 的形式，
    同时优化 coarse 网络和 fine 网络。对于coarse网络，论文采样较为稀疏的$N_c$个点，并将前述式子$\hat{C}(r)$的离散求和函数重新表示为："""
    st.latex("""
    \hat{C}_c(r)=\sum_{i=1}^{N_c}{w_ic_i},w_i=T_i(1-exp(-\sigma_i{\delta_i}))
    """)
    r"""$ \qquad \qquad $ 然后对权值做归一化：$\hat{w}_i=\frac{wi}{\sum_{j=1}^{N_c}{w_j}}$，此处的$\hat{w}_i$可以看作时沿着射线的概率密度函数(PDF)，通过这个概率密度函数可以粗略地得到射线上物体的分布情况。接下来，基于得到的分布使用逆变换采样(inverse transform sampling)来采样$N_f$个点，并用这$N_f$个点和前面的$N_c$个
    点通过前面公式$\hat{C}(r)$一同计算fine网络的渲染结果$\hat{C}_f(r)$。虽然coarse to fine是计算机视觉领域中常见的一个思路，但这篇论文中用coarse网络来生成概率密度很眼熟，再基于概率密度函数采样更精细的点算得上很有趣新颖的做法。采样的图示如下："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片68.png', caption='图68')
    r"""$ \qquad \qquad $ 上图中，白色的点为第一阶段采样的$N_c$个点，而黄色的点为第二阶段根据PDF采样的$N_f$个点。"""
    r"""$ \qquad \qquad \textbf{损失函数：}$ """
    r"""$ \qquad \qquad $ 由于渲染函数是可微的，所以可以通过下式来计算损失："""
    st.latex("""
    L=\sum_{r\in{R}}{[\Vert \hat{C}_c(r)-C(r) \Vert_2^2+ \Vert \hat{C}_f(r)-C(r) \Vert_2^2]}
    """)
    r"""$ \qquad \qquad $ 其中R为每批射线的集合，$C(r)、\hat{C}_c(r)、\hat{C}_f(r)$分别为射线r的ground truth，粗体积(coarse volume)预测、精体积(fine volume)预测的RGB颜色。虽然最终的渲染来自$\hat{C}_f(r)$，但也要同时最小化$\hat{C}_c(r)$的损失，从而使粗网络的权值分布可以用于细网络中的样本分配。"""
    r"""$ \qquad \textbf{代码运行结果：}$ """
    r"""$ \qquad \qquad $ 以下是运行200k轮后生成的lego模型（显卡RTX3060 6G，训练时间约15h），代码使用pytorch版本。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片69.gif', caption='图69')

    """- 三维重建的应用场景：比如增强现实（AR）、混合现实（MR)、机器人导航、自动驾驶等领域的核心技术之一。"""
    r"""$ \qquad $ 影像娱乐："""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./pages/图片/图片44.png', caption='图44')
    r"""$ \qquad $ 智能家居："""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./pages/图片/图片45.png', caption='图45')
    r"""$ \qquad $ 文物重建、AR旅游"""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./pages/图片/图片46.png', caption='图46')
    r"""$ \qquad $ 自动驾驶"""
    r"""$ \qquad \qquad $ 在自动驾驶领域的主要应用是高精地图的构建，此应用对于自动驾驶的算法迭代优化、测试都非常重要。"""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./pages/图片/图片47.png', caption='图47')
    r"""$ \qquad $ 大型场景的构建"""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./pages/图片/图片48.png', caption='图48')
    r"""$ \qquad $ 逆向工程"""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./pages/图片/图片49.png', caption='图49')
    r"""$ \qquad $ 机器人、导航"""
    _, col1, _ = st.columns([1, 2, 1])
    with col1:
        st.image('./pages/图片/图片50.png', caption='图50')

with tab4:
    """- 图像分割的定义："""
    """$ \qquad $ 图像分割是计算机视觉研究中的一个经典难题，已称为图像理解领域关注的一个人点，图像分割是图像分析的第一步，是计算机视觉的基础，是图像理解的重要组成部分，同时也是图像处理中最困难的问题之一。"""
    """$ \qquad $ 所谓图像分割是指根据灰度、彩色、空间纹理、几何形状等特征把图像划分成若干个互不相交的区域，使得这些特征在同一区域内表现出一致性或相似性，而在不同区域间表现出明显的不同。简单的说就是在一副图像中，把目标从背景中分离出来。"""
    """$ \qquad $ 对于灰度图像来说，区域内部的像素一般具有灰度相似性，而在区域的边界上一般具有灰度不连续性。"""
    """$ \qquad $ 关于图像分割技术，由于问题本身的重要性和困难性，从20世纪70年代起图像分割问题就吸引了很多研究人员为之付出了巨大的努力。虽然到目前为止，还不存在一个通用的完美的图像分割的方法，但是对于图像分割的一般性规律则基本上已经达成的共识，已经产生了相当多的研究成果和方法。"""
    """- 传统分割方法："""
    """$ \qquad $ 传统分割方法是深度学习大火之前人们利用数字图像处理、拓扑学、数学等方面的知识来进行图像分割的方法。传统分割方法有很多思想值得我们学习。"""
    """$ \qquad $ 1.基于阈值的分割方法"""
    """$ \qquad \qquad $ 阈值法的基本思想是基于图像的灰度特征来计算一个或多个灰度阈值，并将图像中每个像素的灰度值与阈值作比较，最后将像素根据比较结果分到合适的类别中。因此，该方法最关键的一步是按照某个准则函数来求解最佳灰度阈值。"""
    """$ \qquad \qquad $ 阈值法特别适用于目标和背景占据不同灰度级范围的图。"""
    """$ \qquad \qquad $ 图像若只有目标和背景两大类，那么只需要选取一个阈值进行分割，此方法成为单阈值分割；但是如果图像中有多个目标需要提取，单一阈值的分割就会出现作物，在这种情况下就需要选取多个阈值将每个目标分隔开，这种分割方法相应的成为多阈值分割。"""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片70.png', caption='图70')
    """$ \qquad $ 阈值分割方法的优缺点："""
    """$ \qquad \qquad $ 计算简单，效率较高；"""
    """$ \qquad \qquad $ 只考虑像素点灰度值本身的特征，一般不考虑空间特征，因此对噪声比较敏感，鲁棒性不高。
从前面的介绍里我们可以看出，阈值分割方法的最关键就在于阈值的选择。若将智能遗传算法应用在阀值筛选上，选取能最优分割图像的阀值，这可能是基于阀值分割的图像分割法的发展趋势。"""
    """$ \qquad \\textbf{2.基于区域的图像分割方法：}$ """
    """$ \qquad \qquad $ 基于区域的分割方法是以直接寻找区域为基础的分割技术，基于区域提取方法有两种基本形式：一种是区域生长，从单个像素出发，逐步合并以形成所需要的分割区域；另一种是从全局出发，逐步切割至所需的分割区域。"""
    """$ \qquad \\textbf{3.基于边缘检测的分割方法：} $ """
    """$ \qquad \qquad $ 基于边缘检测的图像分割算法试图通过检测包含不同区域的边缘来解决分割问题。它可以说是人们最先想到也是研究最多的方法之一。通常不同区域的边界上像素的灰度值变化比较剧烈，如果将图片从空间域通过傅里叶变换到频率域，边缘就对应着高频部分，这是一种非常简单的边缘检测算法。"""
    """$ \qquad \qquad$ 边缘检测技术通常可以按照处理的技术分为串行边缘检测和并行边缘检测。串行边缘检测是要想确定当前像素点是否属于检测边缘上的一点，取决于先前像素的验证结果。并行边缘检测是一个像素点是否属于检测边缘高尚的一点取决于当前正在检测的像素点以及与该像素点的一些临近像素点。"""
    """- 结合特定工具的图像分割方法："""
    """$ \qquad \\textbf{1.基于小波分析和小波变换的图像分割方法：}$ """
    """$ \qquad \qquad $ 小波变换是近年来得到的广泛应用的数学工具，也是现在数字图像处理必学部分，它在时间域和频率域上都有量高的局部化性质，能将时域和频域统一于一体来研究信号。"""
    """$ \qquad \qquad $ 小波变换具有多尺度特性能够在不同尺度上对信号进行分析，因此在图像分割方面得到了应用。"""
    """$ \qquad \\textbf{2.基于遗传算法的图像分割：}$ """
    """$ \qquad \qquad $ 遗传算法（Genetic Algorithms，简称GA）是1973年由美国教授Holland提出的，是一种借鉴生物界自然选择和自然遗传机制的随机化搜索算法。是仿生学在数学领域的应用。"""
    """$ \qquad \qquad $ 其基本思想是，模拟由一些基因串控制的生物群体的进化过程，把该过程的原理应用到搜索算法中，以提高寻优的速度和质量。此算法的搜索过程不直接作用在变量上，而是在参数集进行了编码的个体，这使得遗传算法可直接对结构对象（图像）进行操作。整个搜索过程是从一组解迭代到另一组解，采用同时处理群体中多个个体的方法，降低了陷入局部最优解的可能性，并易于并行化。搜索过程采用概率的变迁规则来指导搜索方向，而不采用确定性搜索规则，而且对搜索空间没有任何特殊要求（如连通性、凸性等），只利用适应性信息，不需要导数等其他辅助信息，适应范围广。"""
    """$ \qquad \\textbf{3.基于主动轮廓模型的分割方法：}$ """
    """$ \qquad \qquad $ 主动轮廓模型（active contours）是图像分割的一种重要方法，具有统一的开放式的描述形式，为图像分割技术的研究和创新提供了理想的框架。在实现主动轮廓模型时，可以灵活的选择约束力、初始轮廓和作用域等，以得到更佳的分割效果，所以主动轮廓模型方法受到越来越多的关注。"""
    """$ \qquad \qquad $ 该方法是在给定图像中利用曲线演化来检测目标的一类方法，基于此可以得到精确的边缘信息。其基本思想是，先定义初始曲线C，然后根据图像数据得到能量函数，通过最小化能量函数来引发曲线变化，使其向目标边缘逐渐逼近，最终找到目标边缘。这种动态逼近方法所求得的边缘曲线具有封闭、光滑等优点。"""
    """- 基于深度学习的分割："""
    """$ \qquad $ 1.基于特征编码(feature encoder based)"""
    """$ \qquad \qquad $ 在特征提取领域中VGGnet和ResNet是两个非常有统治力的方法。"""
    """$ \qquad $ 2.基于区域选择(regional proposal based)"""
    """$ \qquad \qquad $ Regional proposal 在计算机视觉领域是一个非常常用的算法，尤其是在目标检测领域。其核心思想就是检测颜色空间和相似矩阵，根据这些来检测待检测的区域。然后根据检测结果可以进行分类预测。
在语义分割领域，基于区域选择的几个算法主要是由前人的有关于目标检测的工作渐渐延伸到语义分割的领域的。"""
    """$ \qquad \qquad $ Stage I: R-CNN"""
    """$ \qquad \qquad $ Stage II: Fast R-CNN"""
    """$ \qquad \qquad $ Stage III: Faster R-CNN"""
    """$ \qquad \qquad $ Stage IV: Mask R-CNN"""
    """$ \qquad \qquad $ Stage V: Mask Scoring R-CNN"""
    """$ \qquad $ 3.基于RNN的图像分割："""
    """$ \qquad \qquad $ Recurrent neural networks（RNNs）除了在手写和语音识别上表现出色外，在解决计算机视觉的任务上也表现不俗，在本篇文章中我们就将要介绍RNN在2D图像处理上的一些应用，其中也包括介绍使用到它的结构或者思想的一些模型。"""
    """$ \qquad \qquad $ RNN是由Long-Short-Term Memory（LSTM）块组成的网络，RNN来自序列数据的长期学习的能力以及随着序列保存记忆的能力使其在许多计算机视觉的任务中游刃有余，其中也包括语义分割以及数据标注的任务。接下来的部分我们将介绍几个使用到RNN结构的用于分割的网络结构模型："""
    """$ \qquad \qquad $ 1.ReSeg模型"""
    """$ \qquad \qquad $ 2.MDRNNs(Multi-Dimensional Recurrent Neural Networks)模型"""
    """$ \qquad $ 4.基于上采样/反卷积的分割方法"""
    """$ \qquad \qquad $ 卷积神经网络在进行采样的时候会丢失部分细节信息，这样的目的是得到更具特征的价值。但是这个过程是不可逆的，有的时候会导致后面进行操作的时候图像的分辨率太低，出现细节丢失等问题。因此我们通过上采样在一定程度上可以不全一些丢失的信息，从而得到更加准确的分割边界。
接下来介绍几个非常著名的分割模型："""
    """$ \qquad \qquad $ a.FCN(Fully Convolutional Network)"""
    """$ \qquad \qquad $ b.SetNet"""
    """$ \qquad $ 5.基于提高特征分辨率的方法"""
    """$ \qquad \qquad $ 恢复在深度卷积神经网络中下降的分辨率，从而获取更多的上下文信息。"""
    """$ \qquad \qquad $ 应用：Google提出的DeepLab。"""
    """$ \qquad $ 6.基于特征增强的分割方法"""
    """$ \qquad \qquad $ 基于特征增强的分割方法包括：提取多尺度特征或者从一系列嵌套的区域中提取特征。在图像分割的深度网络中，CNN经常应用在图像的小方块上，通常称为以每个像素为中心的固定大小的卷积核，通过观察其周围的小区域来标记每个像素的分类。在图像分割领域，能够覆盖到更大部分的上下文信息的深度网络通常在分割的结果上更加出色，当然这也伴随着更高的计算代价。多尺度特征提取的方法就由此引进。"""
    """$ \qquad \qquad $ 应用：SLIC:simple linear iterative cluster的生成超像素的算法。"""
    """$ \qquad $ 7.使用CRF/MRF的方法"""
    """$ \qquad \qquad $ MRF其实是一种基于统计的图像分割算法，马尔可夫模型是指一组事件的集合，在这个集合中，事件逐个发生，并且下一刻事件的发生只由当前发生的事件决定，而与再之前的状态没有关系。而马尔可夫随机场，就是具有马尔可夫模型特性的随机场，就是场中任何区域都只与其临近区域相关，与其他地方的区域无关，那么这些区域里元素（图像中可以是像素）的集合就是一个马尔可夫随机场。"""
    """$ \qquad \qquad $ CRF的全称是Conditional Random Field，条件随机场其实是一种特殊的马尔可夫随机场，只不过是它是一种给定了一组输入随机变量X的条件下另一组输出随机变量Y的马尔可夫随机场，它的特点是埃及设输出随机变量构成马尔可夫随机场，可以看作是最大熵马尔可夫模型在标注问题上的推广。"""



    """- U-Net:"""
    """$ \qquad $ Unet最早发表在2015的MICCAI上，而后成为大多数医疗影像语义分割任务的baseline,也启发了大量研究者去思考U型语义分割网络。"""
    """$ \qquad $ Unet的结构的两大特点，U型结构和skip-connection(如下图)："""
    _, col1, _ = st.columns([1, 12, 1])
    with col1:
        st.image('./pages/图片/图片60.png', caption='图60')

with tab5:
    """- 背景："""
    """$ \qquad $ 随着各种数字仪器和数码产品的普及，图像和视频已成为人类活动中最常用的信息载体，它们包含着物体的大量信息，成为人们获取外界原始信息的主要途径。然而在图像的获取、传输和存贮过程中常常会受到各种噪声的干扰和影响而使图像降质，并且图像预处理算法的好坏又直接关系到后续图像处理的效果，如图像分割、目标识别、边缘提取等，所以为了获取高质量数字图像，很有必要对图像进行降噪处理，尽可能的保持原始信息完整性（即主要特征）的同时，又能够去除信号中无用的信息。所以，降噪处理一直是图像处理和计算机视觉研究的热点。"""
    """$ \qquad $ 图像去噪的最终目的是改善给定的图像，解决实际图像由于噪声干扰而导致图像质量下降的问题。通过去噪技术可以有效地提高图像质量，增大信噪比，更好的体现原来图像所携带的信息，作为一种重要的预处理手段，人们对图像去噪算法进行了广泛的研究。在现有的去噪算法中，有的去噪算法在低维信号图像处理中取得较好的效果，却不适用于高维信号图像处理；或者去噪效果较好，却丢失部分图像边缘信息，或者致力于研究检测图像边缘信息，保留图像细节。如何在抵制噪音和保留细节上找到一个较好的平衡点，成为近年来研究的重点。"""
    """- 图像噪声概念："""
    """$ \qquad $ 噪声可以理解为“妨碍人们感觉器官对所接收的信源信息理解的因素”。例如，一幅黑白图片，其平面亮度分布假定为f(x，y)，那么对其接收起干扰作用的亮度分布R(x，y)，即可称为图像噪声。但是，噪声在理论上可以定义为“不可预测，只能用概率统计方法来认识的随机误差”。因此将图像噪声看成是多维随机过程是合适的，因而描述噪声的方法完全可以借用随机过程的描述，即用其概率分布函数和概率密度分布函数。但在很多情况下，这样的描述方法是很复杂的，甚至是不可能的。而实际应用往往也不必要。通常是用其数字特征，即均值方差，相关函数等。因为这些数字特征都可以从某些方面反映出噪声的特征。"""
    """- 常见的噪声图像："""
    """$ \qquad $ 1.加性噪声"""
    """$ \qquad \qquad $ 加性嗓声和图像信号强度是不相关的，如图像在传输过程中引进的“信道噪声”电视摄像机扫描图像的噪声的。"""
    """$ \qquad $ 2.乘性噪声"""
    """$ \qquad \qquad $ 乘性嗓声和图像信号是相关的，往往随图像信号的变化而变化，如飞点扫描图像中的嗓声、电视扫描光栅、胶片颗粒造成等。"""
    """$ \qquad $ 3.量化噪声"""
    """$ \qquad \qquad $ 量化嗓声是数字图像的主要噪声源，其大小显示出数字图像和原始图像的差异，减少这种嗓声的最好办法就是采用按灰度级概率密度函数选择化级的最优化措施。"""
    """$ \qquad $ 4.“椒盐”噪声"""
    """$ \qquad \qquad $ 此类嗓声如图像切割引起的即黑图像上的白点，白图像上的黑点噪声，在变换域引入的误差，使图像反变换后造成的变换噪声等。"""
    """- 图像噪声模型："""
    """$ \qquad $ 实际获得的图像含有的噪声，根据不同分类可将噪声进行不同的分类。从噪声的概率分情况来看，可分为高斯噪声、瑞利噪声、伽马噪声、指数噪声和均匀噪声。"""

    """- 图像去噪——DnCNN:"""
    """$ \qquad $ DnCNN(Denoising Convolutional Neural Network)顾名思义，就是用于去噪的卷积神经网络，是图像去噪领域的经典文章。"""
    """$ \qquad $ 数字图像在数字化和传输过程中，常受到成像设备与外部环境噪声干扰等影像，引入了不同类型的复杂噪声。图像的去噪任务要求在尽可能去除图像中噪声的同时，还应保持原有图像的边缘、纹理等细节结构信息。对于普遍存在的图像模糊问题，如何有效估计模糊过程、处理噪声和估计误差等，将对恢复高质量、清晰的图像至关重要。"""
    """$ \qquad $ DnCNN提出使用卷积通过端到端的残差学习，从函数回归角度用卷积神经网络将噪声从噪声图像中分离出来，取得了显著优于其他方法的去噪结果。"""
    _, col1, _ = st.columns([1, 12, 1])
    with col1:
        st.image('./pages/图片/图片61.png', caption='图61')
    """- 图像超分辨率："""
    """$ \qquad $ 超分辨率的定义：图像超分辨率（super-resolution,SR）是指利用算法将图像从低分辨率（low resolution,LR）恢复到高分辨率（high resolution,HR）的过程，是计算机视觉和图像的重要技术之一。图像超分辨率技术根据其输入输出不同大致可分为三类，即多图像、视频和单图像超分辨率。"""
    """$ \qquad $ 传统图像超分辨率方法："""
    _, col1, _ = st.columns([1, 4, 1])
    with col1:
        st.image('./pages/图片/图片71.png', caption='图71')
    """$ \qquad \\textbf{基于深度学习的单图像超分辨率重建：}$ """
    """$ \qquad $ 由于深度学习在其他计算机视觉领域中取得了突破性进展，人们尝试引入深度神经网络，通过构建深层次的网络进行训练来解决图像超分辨率重建问题。"""
    """$ \qquad $ 目前，越来越多具有深度学习功能的超分辨率模型被提出，根据是否依赖于LR图像和对应的HR图像训练网络模型，可以粗略地将其分为有监督的超分辨率和无监督的超分辨率，由于有监督的超分辨率技术能够取得较好的重建效果，是目前研究的主流方向。"""
    """$ \qquad $ 1.有监督学习的单图像超分辨率"""
    """$ \qquad \qquad \\textbf{（1）网络模型框架}$"""
    """$ \qquad \qquad $ 单图像超分辨率是从LR到HR空间的一对多映射，由于其不适定性，如何进行上采样，即从低分辨率输入产生高分辨率输出是关键问题。"""
    """$ \qquad \qquad $ a.预上采样超分辨率：预定义的传统算法完成了难以进行的上采样任务，深度CNN仅需要细化大致略图，降低学习难度。"""
    """$ \qquad \qquad $ b. 后上采样超分辨率:通过替换预定义的上采样操作在低维空间中执行大部分映射，使计算复杂度和空间复杂度大大降低，并且也带来了相当快的训练速度。"""
    """$ \qquad \qquad $ c. 渐进上采样超分辨率:该框架下的模型基于CNN级联并逐步重建出更高分辨率的图像，通过将困难的任务分解为多个简单的任务，在每个阶段图像被上采样到更高的分辨率并由CNN细化。拉普拉斯金字塔超分辨率网络（Laplacian pyramid super-resolution networks,LapSRN)是典型采用渐进上采样超分辨率框架的网络模型。"""
    """$ \qquad \qquad $ d. 迭代上下采样超分辨率:试图迭代地计算重建误差，然后将其融合以调整HR图像，该框架下的模型可以更好地挖掘LR-HR图像对之间的深层关系，从而提供更高质量的重建结果。"""
    """$ \qquad \qquad \\textbf{（2）网络设计策略}$"""
    """$ \qquad \qquad $ 在超分辨率领域，研究人员在超分辨率框架之上应用各种网络设计策略来构建最终的超分辨率网络模型。大致地可将超分领域目前流行的网络设计策略归纳为以下几类：残差学习、递归学习、密集连接、生成对抗、注意力机制等。"""
    """$ \qquad \qquad $ a. 残差网络（residual network,ResNet)网络的残差学习可以分为局部残差和全局残差，全局残差要求输入图片与重建的目标图片有极大的相关性，通过分析其相关性进行学习。局部残差主要用于缓解由于网络的加深而带来的退化问题，提高其学习能力。"""
    """$ \qquad \qquad $ b.递归学习可用于缓解更深的网络带来的过度拟合和模型复杂的问题。"""
    """$ \qquad \qquad $ c.密集网络（dense network,DenseNet），通过密集块中的信道数量和连接后挤压信道大大减少了参数数量，有效缓解了梯度消失问题。基于密集网络的图像超分辨率（superresolution dense network,SRDenseNet)将密集连接引入超分辨率中，融合低级和高级特征以提供更丰富的信息来重建高质量细节。"""
    """$ \qquad \qquad $ d. 生成对抗 SRGAN首次将生成对抗网络用于图像的超分辨率重建工作中，利用生成对抗学习策略对网络模型进行优化，网络实现了较大放大因子下的图像重建，模型产生了相对较好的重建效果。"""
    """$ \qquad \qquad $ e. 注意力机制 考虑到不同通道之间特征表示的相互依赖性和相互作用，利用通道注意力机制提出了一个深度残差通道注意力网络（deep residual channel attention networks,RCAN），该模型不仅在表达能力上得到了极大提高，而且SR性能也得到了优化。"""

    """$ \qquad \qquad \\textbf{（3）学习策略} $"""
    """$ \qquad \qquad $ a.损失函数 在超分辨率中，损失函数用于测量生成的超分辨率图像和真实HR图像之间的差异，并指导模型优化。在超分的模型中常见的损失函数有像素损失、内容损失、对抗损失等。"""
    """$ \qquad \qquad \qquad $ i：像素损失主要包括L1损失（即平均绝对误差）和L2损耗（即均方误差），像素损失约束产生的超分辨率图像在像素级上与原来的真实的HR图像更加相似。"""
    """$ \qquad \qquad \qquad $ ii：内容损失主要表示为两个图像的特征表示的欧式距离。"""
    """$ \qquad \qquad \qquad $ iii：对抗损失 SRGAN[20]中通过生成器和鉴别器的相互对抗学习产生相当好的输出。"""
    """$ \qquad \qquad $ b. 批量标准化 batch normalization,BN）以减少网络的内部协变量偏移。通过此策略可以避免梯度消失和梯度爆炸，加速网络的收敛，提高网络的泛化能力，优化网络结构。"""
    """$ \qquad \qquad $ c.多监督是指增加多个额外监督模型中的信号用于增强梯度传播并避免梯度消失和爆炸，在实际中往往是在损失函数中添加所需的特定条件来实现的，可以通过反向传播监督信号来达到加强模型的训练效果。"""
    """$ \qquad \qquad $ d. 数据集在深度学习中，数据集也同样发挥着重要的作用。通过增大数据集，可以使得网络学习到更多的图像特征，增强网络模型的性能。提前对数据集的图片进行预处理，增加图片的多样性，在数据增强的帮助下，超分辨率模型的性能可以得到极大的提高。"""

    """$ \qquad $ 2.无监督学习的单图像超分辨率"""
    """$ \qquad \qquad $ 由于实际中提供网络训练的图像通常为非配对图像，因此相比有监督的学习，在真正的现实样例中无监督的学习训练建立的模型更加符合实际。"""
    """$ \qquad \qquad \\textbf{(1) “零样本”超分辨率技术} $ """
    """$ \qquad \qquad $ “零样本”超分辨率技术是第一个基于CNN的无监督超分辨率方法，算法利用了深度学习的强大学习功能，但不依赖于之前的训练。在单个图像中利用信息的内部重现，并在测试时训练一个特定于图像的小型CNN，仅从输入图像本身提取一个样本。因此，它可以适应每个图像的不同设置，如旧照片、噪声图像、生物数据以及获取过程未知或不理想的其他图像。"""
    """$ \qquad \qquad \\textbf{(2)弱监督超分辨率技术} $ """
    """$ \qquad \qquad $ 使用未配对的LR-HR图像学习具有弱监督学习的超分辨率模型，提出了弱监督超分辨率技术。
首先训练HR到LR的GAN，然后使用未配对的LR-HR图像来学习退化，通过应用两阶段过程，模型有效提高了实际LR图像超分辨率重建图像的质量。"""

    """- 图像超分辨的应用---SRCNN:"""
    """$ \qquad $ 图像超分辨率问题的研究是在输入一张低分辨率图像时（low resolution）,如何得到一张高分辨率图像（high resolution）。"""
    """$ \qquad $传统的图像插值算法可以在某种程度上获得这种效果，比如最近邻插值、双线性插值和双三次插值等，但是这些算法获得的高分辨率图像效果并不理想。SRCNN通过卷积算法获得了优秀的高分辨率重建图像，效果如图。"""
    _, col1, _ = st.columns([1, 12, 1])
    with col1:
        st.image('./pages/图片/图片62.png', caption='图62')
    """$ \qquad $ SRCNN是end-to-end(端到端)的超分算法，所以在实际应用中不需要任何人工干预或者多阶段的计算，其网络图如下："""
    _, col1, _ = st.columns([1, 12, 1])
    with col1:
        st.image('./pages/图片/图片63.png', caption='图63')
    """$ \qquad $ 实际上，SRCNN需要一个预处理过程：将输入的低分辨率图像进行bicubic插值（双三次插值）。"""

with tab6:
    """- 机器学习模型分为两大类，一类是判别模型，一类是生成模型。前者是对一个输入数据判断它的类别或者预测一个实际的数值，后者关注的是怎样生成这个数据本身。"""
    """$ \qquad \\textbf{GAN：}$GAN 包含两个部分，第一个部分是是生成器（Generator），另一个是判别器（Discriminator），都是由 MLP 实现的。 """
    """$ \qquad $ 生成器G的任务是生成看起来自然真实的、与原始数据类似、用以骗过判别器的实例。输入时一个随机噪声，通过噪声生成图片。"""
    """$ \qquad $ 判别器D的任务是判断生成器生成的实例是真实的还是伪造的。输入时一张图片，输出为输入图片是真实图片的概率。"""
    """$ \qquad $ GAN的损失函数如下："""
    st.latex("""
    min_Gmax_DV(D,G) = E_{x\sim{p_{data}(x)}}[logD(x)]+E_{z\sim{p_z(z)}}[log(1-D(G(z)))]
    """)
    """$ \qquad \qquad $ 其中，其中，G代表生成器， D代表判别器， x代表真实数据，$p_{data}$代表真实数据概率密度分布，z代表了随机输入数据，该数据是随机高斯噪声。"""
    """$ \qquad $ 从上式可以看出，从判别器D角度来看，判别器D希望能尽可能区分真实样本x和虚假样本G(z)，因此D(x)必须尽可能大，D(G(z))尽可能小，也就是V(D,G)整体尽可能大。从生成器的角度来看，生成器G希望自己生成的虚假数据G(z)可以尽可能骗过判别器D,也就是希望D(G(z))尽可能大，也就是V(D,G)整体尽可能小。GAN的两个模块在训练相互对抗，最后达到全局最优。"""
    _, col1, _ = st.columns([1, 12, 1])
    with col1:
        st.image('./pages/图片/图片64.png', caption='图64')

    """- 风格迁移StyleGAN:"""
    """$ \qquad $ StyleGAN中的“Style”是指数据集中人脸的主要属性，比如人物的姿态等信息，而不是风格转换中的图像风格，这里Style是指人脸的风格，包括了脸型上面的表情、人脸朝向、发型等等，还包括纹理细节上的人脸肤色、人脸光照等方方面面。"""
    """$ \qquad $ StyleGAN用风格（style）来影像人脸的姿态、身份特征等，用噪声（noise）来影像头发丝、皱纹、肤色等细节部分。"""
    _, col1, _ = st.columns([1, 12, 1])
    with col1:
        st.image('./pages/图片/图片65.png', caption='图65')
