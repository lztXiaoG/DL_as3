from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import PalindromeDataset
from lstm import LSTM
from utils import AverageMeter, accuracy
import matplotlib.pyplot as plt


def train(model, data_loader, optimizer, criterion, device, config):
    # TODO set model to train mode
    ############
    model.train()  # Set model to train mode
    ############
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        ############

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_targets = batch_targets.long()

        model.zero_grad()

        # 前向传播
        outputs = model(batch_inputs)
        # 获取最后一个时间步的预测值并应用 softmax
        # y_t = outputs[:, -1]  # 获取最后一个时间步的预测值
        # # 计算交叉熵损失
        # loss = criterion(y_t, batch_targets)
        loss = criterion(outputs, batch_targets)

        # 反向传播

        loss.backward()

        ############
        # the following line is to deal with exploding gradients
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.max_norm)

        ############
        # 更新统计指标
        acc = accuracy(outputs, batch_targets)
        losses.update(loss.item(), batch_inputs.size(0))
        accuracies.update(torch.tensor(acc), batch_inputs.size(0))

        ############
        # Add more code here ...
        if step % 10 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    # TODO set model to evaluation mode
    ############
    model.eval()  # 设置模型为评估模式
    ############
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        ############
        # 前向传播
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)

        acc = accuracy(outputs, batch_targets)
        losses.update(loss.item(), batch_inputs.size(0))
        accuracies.update(acc, batch_inputs.size(0))

        if step % 10 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)
    return losses.avg, accuracies.avg


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    # Initialize the model that we are going to use
    model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes)
    model.to(device)

    # Initialize the dataset and data loader
    dataset = PalindromeDataset(config.input_length, config.data_size)

    actual_data_size = len(dataset)

    # Split dataset into train and validation sets
    #     train_size = int(config.data_size * config.portion_train)
    #     val_size = config.data_size - train_size

    train_size = int(actual_data_size * config.portion_train)
    train_size = min(train_size, actual_data_size)
    val_size = actual_data_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders for training and validation
    train_dloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(config.max_epoch):
        # Train the model for one epoch
        train_loss, train_acc = train(
            model, train_dloader, optimizer, criterion, device, config)

        # Evaluate the trained model on the validation set
        val_loss, val_acc = evaluate(
            model, val_dloader, criterion, device, config)
        # scheduler.step()  # 更新学习率
        print(f"Epoch [{epoch + 1}/{config.max_epoch}]")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    print('Done training.')
    # 删除同名的文件（如果存在）
    if os.path.exists('loss_plot.png'):
        os.remove('loss_plot.png')
    # 删除同名的文件（如果存在）
    if os.path.exists('accuracy_plot.png'):
        os.remove('accuracy_plot.png')
    # 绘制train和eval的loss变化图
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Eval Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')

    # 绘制train和eval的accuracy变化图
    plt.figure()
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Eval Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=5,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float,
                        default=0.0007, help='Learning rate')
    parser.add_argument('--max_epoch', type=int,
                        default=200, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int,
                        default=10000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8,
                        help='Portion of the total dataset used for training')

    config = parser.parse_args()
    # Train the model
    main(config)
