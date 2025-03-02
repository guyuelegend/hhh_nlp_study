# -*- coding:utf-8-*-
import matplotlib.pyplot as plt


def loss_visual(watch: tuple, save_path, wait_time=20):
    with plt.rc_context(rcParams_dic):
        plt.figure(dpi=300)
        plt.plot(range(len(watch)), [l[0] for l in watch], color='r', label="accuracy_rate")
        plt.plot(range(len(watch)), [l[1] for l in watch], color='g', label="loss_value")
        plt.xlabel("epoch time", fontsize=15, fontweight='bold')
        plt.ylabel("acc/loss", fontsize=15, fontweight='bold')
        plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')  # 设置标签大小及加粗
        plt.xticks(fontproperties='Times New Roman', size=15)
        plt.tight_layout()  # 解决绘图时上下标题重叠现象
        plt.legend()

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
        plt.show(block=False)
        plt.pause(wait_time)
        plt.close()


rcParams_dic = {  # plt画图全局参数的设置
    "font.family": "Times New Roman",  # 字体类型
    "font.weight": "bold",  # 字体加粗
    "axes.labelweight": "bold",  # 背景字体加粗
    "axes.facecolor": "snow",  # 背景颜色
    "font.sans-serif": "KaiTi",  # 便于显示中文标签
    "axes.unicode_minus": False,  # 便于显示负号
}
