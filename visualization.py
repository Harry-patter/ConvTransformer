import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


# 此程序用途如下：
# 1 将数据集中高光谱图像与雷达图像用灰度图的形式可视化
# 2 将训练集和测试集的标签分布可视化
# 3 提供可视化训练结果的函数


# 直方图均衡化函数，可视化时可以获得更清晰的图像
def histogram_equalize(x, bins=65536):
    x = (x-x.min())/(x.max()-x.min())
    x_flatten = x.flatten()
    index = (x_flatten*(bins-1)).floor().long()

    # 计算直方图
    hist_counts = torch.histc(x_flatten, bins=bins, min=0, max=1)

    # 计算累积分布
    cdf = hist_counts.cumsum(dim=0) / hist_counts.sum()

    # 将输入图像的像素值映射到 CDF 上，实现直方图均衡化
    output_tensor = cdf[index].reshape_as(x)

    return output_tensor


# 绘制训练结果折线图
def draw_chart(train_acc, test_acc, train_loss, test_loss, path, show=False):
    # 绘制折线图
    plt.figure()

    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)

    np.save('train_acc', train_acc)
    np.save('test_acc', test_acc)
    np.save('train_loss', train_loss)
    np.save('test_loss', test_loss)

    x = np.arange(train_acc.shape[0])

    plt.plot(x, train_acc, label='train acc')
    plt.plot(x, test_acc, label='test acc')
    plt.plot(x, train_loss, label='train loss')
    plt.plot(x, test_loss, label='test loss')

    plt.legend()

    # plt.title()

    plt.savefig(path, dpi=300)

    if show:
        plt.show(block=True)


# 绘制训练结果正误分布图
def draw_truth(predict, label, path, show=False):
    """
    画出分类结果
    :param path:
    :param predict:
    :param label:
    :return:
    """
    wrong = ((predict != label) * (label != 0)) * 255
    right = ((predict == label) * (label != 0)) * 255
    img = torch.stack((wrong, right, torch.zeros_like(label)), dim=-1)
    img = img.cpu().numpy()
    img = Image.fromarray(img.astype(np.uint8), 'RGB')

    img.save(path, format='png')

    if show:
        img.show()

    return img


# 绘制训练结果分割图
def draw_semantic(predict, path=None, show=False):
    """
    画出分割结果
    :param path:
    :param predict:
    :return:
    """
    # 自定义16种颜色展示标签
    # 标签颜色
    color_map = [(0, 0, 0), (235, 51, 36), (255, 253, 85), (161, 250, 79),
                 (119, 67, 66), (58, 6, 3), (240, 134, 80), (80, 127, 128),
                 (159, 252, 253), (126, 132, 247), (129, 128, 73), (116, 27, 124),
                 (127, 130, 187), (128, 128, 128), (0, 12, 123), (234, 63, 247),
                 (255, 255, 255)
                 ]
    height, width = predict.shape
    semantic = Image.new('RGB', (width, height))
    for x in range(width):
        for y in range(height):
            gray_value = predict[y][x]
            semantic.putpixel((x, y), color_map[gray_value])
    if path:
        semantic.save(path, format='png')

    if show:
        semantic.show()

    return semantic


if __name__ == '__main__':
    # 数据集
    names = ['houston', 'trento', 'muffl']

    # 数据集文件夹
    data_dir = 'dataset'

    # 生成图片存放目录
    image_dir = 'data_visualization'

    # 是否显示生成图像
    show = False

    os.makedirs(image_dir, exist_ok=True)

    for name in names:
        # 读取数据
        data = torch.load(os.path.join(data_dir, name + '.pth'))
        hsi, lidar, train_label, test_label = data['hsi'], data['lidar'], data['train'], data['test']

        # 标签分布
        draw_semantic(train_label, os.path.join(image_dir, name+'train.png'), show)
        draw_semantic(test_label, os.path.join(image_dir, name + 'test.png'), show)
        draw_semantic(train_label+test_label, os.path.join(image_dir, name + 'label.png'), show)

        # hsi可视化
        # 将每个通道数据直方图均衡化
        for i in range(hsi.shape[1]):
            hsi[0, i] = histogram_equalize(hsi[0, i], 66535)
            ...

        # 通过均值转换成灰度图
        hsi = hsi.mean(dim=1)[0]

        hsi = histogram_equalize(hsi)

        # 反归一化
        hsi = hsi * 255

        # 转换成numpy
        hsi = hsi.numpy()

        image = Image.fromarray(hsi.astype(np.uint8))
        # image.show()

        image.save(os.path.join(image_dir, name+'hsi.png'))

        # lidar可视化
        for i in range(lidar.shape[1]):
            lidar = histogram_equalize(lidar)

        # 求均值
        lidar = lidar.mean(dim=1)[0]

        lidar = histogram_equalize(lidar)

        # 反归一化
        lidar = lidar * 255

        # 转换成numpy
        lidar = lidar.numpy().astype(np.uint8)

        image = Image.fromarray(lidar)
        # image.show()
        image.save(os.path.join(image_dir, name+'lidar.png'))


