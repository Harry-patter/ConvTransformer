from ConvTransformer import Network
from visualization import draw_semantic, draw_chart, draw_truth

import argparse
import random
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import os
import matplotlib.pyplot as plt
# 不展示警告信息
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 绘图显示中文


# 归一化和标准化数据
# 因为数据集初始格式不同，有的经过了标准化，有的经过了归一化，有的时原始数据
def normalization(x: torch.tensor, max_min: bool = True, z_score: bool = False):
    """
    标准化数据
    :param x: 输入tensor
    :param max_min: 归一化
    :param z_score: 标准化
    :return:
    """
    for i in range(x.shape[1]):
        if max_min:
            x[0, i] = (x[0, i] - x[0, i].min()) / (x[0, i].max() - x[0, i].min())
        if z_score:
            x[0, i] = (x[0, i] - x[0, i].mean()) / x[0, i].std()


# 随机翻折和旋转图像，实测对结果没太大影响
class RandomRotFlip:
    def __init__(self, rand_start=0):
        self.rand_start = rand_start
        self.k = random.randint(rand_start, rand_start + 26) % 28

    def update(self):
        self.k = random.randint(self.rand_start + 1, self.rand_start + 27)

    def execute(self, x):
        return rot_flip(x, self.k)

    def reverse(self, x):
        return rot_flip_re(x, self.k)


def rot_flip(x, i):
    i = i % 28  # i: 0,27
    flip = i // 4  # flip: 0,3
    rot = i % 7 - 3  # rot: -3:3
    if flip == 0:
        x = x
    elif flip == 1:
        x = torch.flip(x, dims=(-1,))
    elif flip == 2:
        x = torch.flip(x, dims=(-2,))
    else:
        x = torch.flip(x, dims=(-1, -2))

    x = torch.rot90(x, k=rot, dims=(-2, -1,))

    return x


def rot_flip_re(x, i):
    i = i % 28  # i: 0,27
    flip = i // 4  # flip: 0,3
    rot = i % 7 - 3  # rot: -3:3

    x = torch.rot90(x, k=-rot, dims=(-2, -1,))

    if flip == 0:
        x = x
    elif flip == 1:
        x = torch.flip(x, dims=(-1,))
    elif flip == 2:
        x = torch.flip(x, dims=(-2,))
    else:
        x = torch.flip(x, dims=(-1, -2))

    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parser")
    # 程序模式，训练或测试
    parser.add_argument('--mode', choices=['train', 'test'], default='train'
                        , help='train or test')

    # 选择的数据集
    parser.add_argument('--dataset', choices=['houston', 'trento', 'muffl'], default='muffl'
                        , help='choose dataset')

    # 训练时是否加载参数
    parser.add_argument('--load_parm', action='store_true', default=False,
                        help='if load parm of model before train')

    # 训练批次
    parser.add_argument('--epoch', type=int, default=30,
                        help='train epoch')

    # 优化器参数
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='strength of noise')

    # 权重衰减
    parser.add_argument('--weight_decay', type=float, default=.0,
                        help='weight decay of optimizer')

    # 学习率变化步长
    parser.add_argument('--step_size', type=int, default=200,
                        help='weight decay of optimizer')

    # 学习率衰减
    parser.add_argument('--gamma', type=float, default=.1,
                        help='weight decay of optimizer')
    # 随机种子
    parser.add_argument('--seed', type=int, default=None,
                        help='number of seed')

    # 数据集文件夹
    parser.add_argument('--dataset_dir', dest='dataset_dir', type=str, default='dataset',
                        help='directory to save the trained model')

    # 模型文件夹
    parser.add_argument('--model_dir', type=str, default='new_model',
                        help='directory to save the trained model')

    # 生成的模型的存放文件夹
    parser.add_argument('--model_save_dir', type=str, default='new_model',
                        help='directory to save the trained model')

    # 生成的折线图的存放文件夹
    parser.add_argument('--chart_save_dir', type=str, default='chart',
                        help='directory to save the chart')

    # 生成的真值图的存放文件夹
    parser.add_argument('--truth_img_save_dir', type=str, default='truth_img',
                        help='directory to save the truth image')

    # 生成的分割图的存放文件夹
    parser.add_argument('--semantic_img_save_dir', type=str, default='semantic_img',
                        help='directory to save the semantic image')

    args = parser.parse_args()

    # 载入参数
    mode = args.mode
    dataset_name = args.dataset
    load_param = args.load_parm or (mode == 'test')
    epoch = args.epoch if mode == 'train' else 0
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    step_size = args.step_size
    gamma = args.gamma
    seed = args.seed
    dataset_dir = args.dataset_dir
    model_dir = args.model_dir
    model_save_dir = args.model_save_dir
    chart_save_dir = args.chart_save_dir
    truth_img_save_dir = args.truth_img_save_dir
    semantic_img_save_dir = args.semantic_img_save_dir

    # 消除未知类型警告
    dataset_name: str
    dataset_dir: str
    model_dir: str
    model_save_dir: str
    chart_save_dir: str
    truth_img_save_dir: str
    semantic_img_save_dir: str

    # 创建目录
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(chart_save_dir, exist_ok=True)
    os.makedirs(truth_img_save_dir, exist_ok=True)
    os.makedirs(semantic_img_save_dir, exist_ok=True)

    # 设置随机种子
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    # 加载数据集
    dataset_path = os.path.join(dataset_dir, dataset_name + '.pth')
    dataset = torch.load(dataset_path)

    # 加载数据
    hsi = dataset['hsi']
    lidar = dataset['lidar']
    train_label = dataset['train']
    test_label = dataset['test']
    tags = dataset['tag']

    # 统计数据集数据
    cls_num = train_label.max().item()
    train_num, test_num = torch.count_nonzero(train_label).item(), torch.count_nonzero(test_label).item()
    train_index, test_index = torch.where(train_label != 0), torch.where(test_label != 0)

    # 归一化数据
    normalization(hsi, max_min=True, z_score=True)
    normalization(lidar, max_min=True, z_score=True)

    # 模型参数设定
    feature, unet_layer, scalar, attn_layer, head, hidden_dim, = 64, 4, 2, 0, 1, None
    if dataset_name == 'houston':
        feature, unet_layer, scalar, attn_layer, head, hidden_dim, = 64, 4, 2, 0, 16, None
    elif dataset_name == 'trento':
        feature, unet_layer, scalar, attn_layer, head, hidden_dim, = 64, 3, 3, 0, 16, None
    elif dataset_name == 'muffl':
        feature, unet_layer, scalar, attn_layer, head, hidden_dim, = 256, 3, 3, 0, 16, None

    # 生成模型
    model = Network(
        hsi.shape[1], lidar.shape[1], cls_num,
        feature=feature, unet_layer=unet_layer, scalar=scalar, attn_layer=attn_layer, head=head, hidden_dim=hidden_dim
        )
    model = model.cuda()

    # 生成优化器
    param_list = [
        {'params': model.parameters(), 'lr': learning_rate, 'weight_decay': weight_decay},
    ]
    optimizer = torch.optim.Adam(param_list)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 加载模型和优化器参数
    if load_param:
        model_path = os.path.join(model_dir, dataset_name + '.pth')
        checkpoint = torch.load(model_path, map_location='cuda')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 混合精度梯度缩放
    scaler = GradScaler()

    # 损失函数
    criterion = nn.CrossEntropyLoss().cuda()

    # 数据记录
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # 开始训练
    model.train()
    for e in range(epoch):
        print('第{}轮训练开始'.format(e + 1))

        # 生成一个随机的翻折旋转变换，用于弱数据增强，与上一次变换不相同
        rand_opt = RandomRotFlip(e)
        hsi_input = rand_opt.execute(hsi)
        lidar_input = rand_opt.execute(lidar)

        # 混合精度训练
        with autocast():
            # 模型向前传播
            outputs = model(hsi_input.cuda(), lidar_input.cuda())
            # 还原随机翻折旋转
            outputs = rand_opt.reverse(outputs)
            # 转置输出
            outputs = outputs[0].permute(1, 2, 0)

            # 得到分类结果,转换成cpu存储
            result = (outputs.argmax(-1) + 1).cpu()

            # 得到训练损失
            train_loss = criterion(outputs[train_index], train_label[train_index].cuda() - 1)
            print('训练集损失：{:.8f}'.format(train_loss.item()))
            train_losses.append(train_loss.item())

            # 统计训练集分类准确率
            train_right = (result[train_index] == train_label[train_index]).sum().item()
            train_accuracy = train_right / train_num
            train_accuracies.append(train_accuracy)

            # 测试集损失
            test_loss = criterion(outputs[test_index], test_label[test_index].cuda() - 1)
            print('测试集损失：{:.8f}'.format(test_loss.item()))
            test_losses.append((test_loss.item()))

            # 测试集总体分类正确率
            test_right = (result[test_index] == test_label[test_index]).sum().item()
            test_accuracy = test_right / test_num
            test_accuracies.append(test_accuracy)

        # 梯度累计
        scaler.scale(train_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # # 优化参数
        # optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        scheduler.step()

    # 模型测试
    model.eval()
    with torch.no_grad():
        # 模型向前传播
        outputs = model(hsi.cuda(), lidar.cuda())
        outputs = outputs[0].permute(1, 2, 0)

        # 得到分类结果
        result = (outputs.argmax(-1) + 1).cpu()

        # 训练集损失
        train_loss = criterion(outputs[train_index], (train_label[train_index] - 1).cuda())
        train_losses.append(train_loss.item())

        # 训练集正确率
        train_right = (result[train_index] == train_label[train_index]).sum().item()
        train_accuracy = train_right / train_num
        train_accuracies.append(train_accuracy)

        # 训练集分类正确率
        for i, tag in enumerate(tags):
            # 每一类分类正确个数
            tag_right = (result[train_label == (i + 1)] == train_label[train_label == (i + 1)]).sum().item()

            # 每一类数量
            tag_num = torch.count_nonzero(train_label == (i + 1)).item()
            # print(tag_num)

            # 每一类准确率
            tag_accuracy = tag_right / tag_num
            print('{}:{:.3f}%'.format(tag, tag_accuracy * 100))
        print('训练集总体准确率{:.3f}%'.format(train_accuracy * 100))

        # 测试集损失
        test_loss = criterion(outputs[test_index], (test_label[test_index] - 1).cuda())
        test_losses.append((test_loss.item()))

        # 测试集正确率
        test_right = (result[test_index] == test_label[test_index]).sum().item()
        test_accuracy = test_right / test_num
        test_accuracies.append(test_accuracy)

        # 测试集分类正确率
        tag_accuracies = []
        Na_Np = 0
        for i, tag in enumerate(tags):
            # 每一类分类正确个数
            tag_right = (result[test_label == (i + 1)] == test_label[test_label == (i + 1)]).sum().item()

            # 每一类个数
            tag_num = torch.count_nonzero(test_label == (i + 1)).item()

            # 每一类准确率
            tag_accuracy = tag_right / tag_num
            Na_Np += (tag_right * tag_num)
            tag_accuracies.append(tag_accuracy)

            # print('{}:{:.3f}%'.format(tag, tag_accuracy * 100))
            # 为了方便录入表格
            print('{:.3f}'.format(tag_accuracy * 100))
        # print('AA{:.3f}%'.format(test_accuracy * 100))
        # 为了方便录入表格
        print('{:.3f}'.format(test_accuracy * 100))

        OA = np.array(tag_accuracies).mean()
        # print('OA:', OA)
        print('{:.3f}'.format(OA * 100))
        Pe = Na_Np / test_right ** 2
        kappa = (OA - Pe) / (1 - Pe)
        # print('kappa:', kappa)
        # 为了方便录入表格
        print('{:.3f}'.format(kappa * 100))

        if mode == 'train':
            # 保存模型参数
            model_save_path = os.path.join(model_save_dir, dataset_name + '.pth')
            model = model.to('cpu')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_save_path)

            # 绘制折线图
            chart_path = os.path.join(chart_save_dir, dataset_name + '.png')
            draw_chart(train_accuracies, test_accuracies, train_losses, test_losses, chart_path)

            # 绘制真值图
            train_img_path = os.path.join(truth_img_save_dir, dataset_name + '_train.png')
            draw_truth(result, train_label, train_img_path)

            test_img_path = os.path.join(truth_img_save_dir, dataset_name + '_test.png')
            draw_truth(result, test_label, test_img_path)

            # 绘制分割图
            semantic_img_path = os.path.join(semantic_img_save_dir, dataset_name + '.png')
            draw_semantic(result, semantic_img_path)
    ...
