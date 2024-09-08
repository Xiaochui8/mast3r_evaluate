import matplotlib.pyplot as plt
def test_plot():
    # 数据
    x = [1, 2, 3, 4, 5]
    y = [2, 3, 5, 7, 11]

    # 创建图形和坐标轴
    plt.figure()

    # 绘制折线图
    plt.plot(x, y, marker='o', linestyle='-', color='b')

    # 添加标题和标签
    plt.title('简单折线图')
    plt.xlabel('X 轴')
    plt.ylabel('Y 轴')

    # 显示图形
    plt.show()



if __name__ == '__main__':
    test_plot()