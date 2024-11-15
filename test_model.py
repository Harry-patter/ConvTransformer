import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler
from Network.Network import Network

from Tags import dataset_tags

import os

from Draw import draw_truth

if __name__ == '__main__':
    # 选择数据集
    choice = 0

    # 是否加载模型参数
    load_param = True

    # 批次
    epoch = 0

    # 文件夹
    names = ['houston', 'trento', 'muffl']
    data_dir = 'dataset'
    model_dir = 'model'

    # 加载标签名字
    data_name = names[choice]
    tags = dataset_tags[data_name]

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(model_dir, data_name + '.pth')

    chart_dir = 'chart'
    if not os.path.exists(chart_dir):
        os.mkdir(chart_dir)

    truth_img_dir = 'truth img'
    if not os.path.exists(truth_img_dir):
        os.mkdir(truth_img_dir)

    # 读取数据, 数据太大，暂时不放进cuda
    data = torch.load(os.path.join(data_dir, data_name + '.pth'))
    hsi, lidar, train_label, test_label = data['hsi'], data['lidar'], data['train'], data['test']
    train_label, test_label = train_label.cuda(), test_label.cuda()

    if choice != 2:
        # 按通道归一化后标准化
        for i in range(hsi.shape[1]):
            # hsi[0, i] = histogram_equalize(hsi[0, i])
            hsi[0, i] = (hsi[0, i] - hsi[0, i].min()) / (hsi[0, i].max() - hsi[0, i].min())
            # hsi[0, i] = (hsi[0, i] - hsi[0, i].mean()) / hsi[0, i].std()

        for i in range(lidar.shape[1]):
            # lidar[0, i] = histogram_equalize(lidar[0, i])
            lidar[0, i] = (lidar[0, i] - lidar[0, i].min()) / (lidar[0, i].max() - lidar[0, i].min())
            # lidar[0, i] = (lidar[0, i] - lidar[0, i].mean()) / lidar[0, i].std()

    # 统计数据集数据
    cls_num = train_label.max().item()
    train_num, test_num = torch.count_nonzero(train_label).item(), torch.count_nonzero(test_label).item()
    train_index, test_index = torch.where(train_label != 0), torch.where(test_label != 0)

    # 生成模型
    model = Network(hsi.shape[1], lidar.shape[1], cls_num)
    model = model.cuda()

    # 加载模型和优化器参数
    if load_param:
        model_path = os.path.join(model_dir, data_name + '.pth')
        checkpoint = torch.load(model_path, map_location='cuda')
        model.load_state_dict(checkpoint['model'])

    # 损失函数
    criterion = nn.CrossEntropyLoss().cuda()

    # 模型测试
    model.eval()

    scaler = GradScaler()

    with torch.no_grad():
        # 模型向前传播
        outputs = model(hsi.cuda(), lidar.cuda())
        outputs = outputs[0].permute(1, 2, 0)

        # 得到分类结果
        result = outputs.argmax(2) + 1

        # 训练集正确率
        train_right = (result[train_index] == train_label[train_index]).sum().item()
        train_acc = train_right / train_num

        # 训练集分类正确率
        for i, t in enumerate(tags):
            t_right = (result[train_label == (i + 1)] == train_label[train_label == (i + 1)]).sum().item()
            t_num = torch.count_nonzero(train_label == (i + 1)).item()
            print('{}:{:.3f}%'.format(t, t_right * 100 / t_num))
        print('训练集总体准确率{:.3f}%'.format(train_acc * 100))

        # 测试集正确率
        test_right = (result[test_index] == test_label[test_index]).sum()
        test_acc = (test_right / test_num).cpu().numpy()

        # 测试集分类正确率
        t_accs = []
        Na_Np = 0
        for i, t in enumerate(tags):
            t_right = (result[test_label == (i + 1)] == test_label[test_label == (i + 1)]).sum().item()
            t_num = torch.count_nonzero(test_label == (i + 1)).item()
            Na_Np += (t_right*t_num)
            t_accs.append(t_right * 100 / t_num)
            print('{}:{:.3f}%'.format(t, t_right * 100 / t_num))
        print('测试集总体准确率{:.3f}%'.format(test_acc * 100))
        OA = np.array(t_accs).mean()/100
        print('OA:', OA)
        Pe = Na_Np/test_right**2
        kappa = (OA-Pe)/(1-Pe)
        print('kappa:', kappa)

        # 绘制真值图
        train_img_path = os.path.join(truth_img_dir, data_name + '_train.png')
        draw_truth(result, train_label, train_img_path)

        test_img_path = os.path.join(truth_img_dir, data_name + '_test.png')
        draw_truth(result, test_label, test_img_path)
    ...
