import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import config
import os
import sys
from scipy.interpolate import make_interp_spline

def plot_loop(loop):
    # Add the first point to the end of the array to create a loop
    x =loop[:,0]
    y = loop [:,1]

    x = np.append(x, x[0])
    y = np.append(y, y[0])

    # Plot the points and connect them with lines
    plt.plot(x, y, '-')
def plot_smooth_loop(points,ax,kwargs):
    """
    绘制平滑的点云环形曲线

    Args:
        points: ndarray, shape (n_points, 2), 二维点云矩阵

    Returns:
        None
    """
    points = np.append(points, [points[0]], axis=0)
    n_points = points.shape[0]

    # 构造绘制曲线所需的坐标序列
    x = np.linspace(0, 1, n_points)
    x_new = np.linspace(0, 1, 300)
    spl_x = make_interp_spline(x, points[:, 0], k=3)
    spl_y = make_interp_spline(x, points[:, 1], k=3)
    y_new = spl_y(x_new)
    x_new = spl_x(x_new)

    # 绘制曲线
    ax.plot(x_new, y_new, **kwargs)

    # 添加起点和终点，并将曲线连成环
    #ax.plot([x_new[-1], x_new[0]], [y_new[-1], y_new[0]], **kwargs)

    # 隐藏坐标轴
    ax.axis('off')
    ax.legend(loc=3,prop={'size':10})


def stiefel_rotate(V1, V2):
    """
    将Stiefel流形V1上的元素通过正交矩阵变换旋转到Stiefel流形V2上的元素
    :param V1: numpy.ndarray, shape=[n, m]
    :param V2: numpy.ndarray, shape=[n, m]
    :return: numpy.ndarray, shape=[m, m]
    """
    Q1, R1 = np.linalg.qr(V1)
    Q2, R2 =  np.linalg.qr(V2)
    U, _, VT = np.linalg.svd(np.dot(Q1.T, Q2))
    return np.dot(U, VT)

sys.path.append(".")
sys.path.append(os.getcwd())
fold_read = config.foldname
print(fold_read)
#fold_write = config.fold_strong
np.random.seed(42)
X = np.load(fold_read + 'age_test.npy')
Y = np.load(fold_read + 'data_test.npy')
v = np.load(fold_read + 'v.npy')
p = np.load(fold_read + 'p.npy')

str = ['oogd','aoogd','rceg','arceg','ogd','groud']
kw_dict ={}
kw_dict['oogd'] = {'c': '#1772b2', 'ls' : '-' , 'linewidth': 2,'label':'R-OOGD'}
kw_dict['aoogd'] = {'c': '#249c24', 'ls': '-', 'linewidth': 2,'label':'R-AOOGD'}
kw_dict['rceg'] = {'c': '#ff7f0e', 'ls': '-', 'linewidth': 2,'label':'RARDv-exp'}
kw_dict['arceg'] = {'c':'#d62425','ls': '-', 'label':'RARDv','linewidth':2}
kw_dict['ogd'] =  {'c':'#6a5acd','ls': '-', 'label':'R-OGD','linewidth':2}
kw_dict['ground'] =  {'c':'#6a5a2d','ls': '-', 'label':'True','linewidth':2}
# 生成随机的点
N = X.shape[0]
for i in range(N):
    ground = Y[i]
    age = int(X[i]*20+75)
    fig, ax = plt.subplots()
    for j in range (5):
        predict = config.G.exp(p[j],X[i]*v[j])
        #predict = predict
        #predict[:,0] = -predict[:,0] 
        #predict[:,1] = -predict[:,1] 
        predict = predict @ stiefel_rotate(predict,ground)
        if predict[0,0] * ground [0,0] <0:
            predict[:,0] = -predict[:,0] 
        if predict[0,1] * ground [0,1] <0:
            predict[:,1] = -predict[:,1] 
        plot_smooth_loop(predict,ax,kw_dict[str[j]])
        
    plot_smooth_loop(ground,ax,kw_dict['ground'])
 
    plt.show()
    plt.close()
