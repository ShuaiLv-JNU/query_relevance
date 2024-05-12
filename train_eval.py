# coding: UTF-8
import os
import torch
from sklearn import metrics
import time
from torch.utils.data import DataLoader
import copy
import logging
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.autograd import Variable
from utils import convert_examples_to_features, BuildDataSet
logger = logging.getLogger(__name__)


"""
模型训练主函数
:param config: 配置信息
:param model: 模型
:param tokenizer: 分词器
:param train_data: 训练集数据，list形式
:param dev_data: 验证集数据，list形式
"""
def train(
    config,
    model,
    tokenizer,
    train_data=None,
    dev_data=None,
):
    dev_acc = 0.
    # 加载模型，将模型拷贝到config指定的gpu
    model_example = copy.deepcopy(model).to(config.device)
    best_model = None

    if train_data:

        config.train_num_examples = len(train_data)
        # 将训练数据转换为模型接收的InputFeatures
        train_features = convert_examples_to_features(
            examples=train_data,
            tokenizer=tokenizer,
            max_length=config.pad_size,
            data_type='train'
        )
        train_dataset = BuildDataSet(train_features)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        # 加载验证集数据
        if dev_data is not None:
            config.dev_num_examples = len(dev_data)
            dev_features = convert_examples_to_features(
                examples=dev_data,
                tokenizer=tokenizer,
                max_length=config.pad_size,
                data_type='dev'
            )
            dev_dataset = BuildDataSet(dev_features)
            dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
        else:
            dev_loader = None

        model_train(config, model_example, train_loader, dev_loader)

"""
模型训练函数，主要包括设置优化器，学习率衰减策略，训练过程中验证，保存最优模型等
:param config: 配置信息
:param model: 模型
:param train_iter: 训练集迭代器
:param dev_iter: 验证集迭代器，可为None
"""
def model_train(config, model, train_iter, dev_iter=None):
    start_time = time.time()

    # Prepare optimizer and schedule (linear warmup and decay)
    # 配置AdamW优化器，对不同的参数设置不同的学习率和正则化策略
    no_decay = ["bias", "LayerNorm.weight"]# 不需要进行权重衰减的参数名称
    diff_part = ["bert.embeddings", "bert.encoder"]# 模型的特定部分
    # 分成两组
    if config.diff_learning_rate is False:
        optimizer_grouped_parameters = [
            # 需要weight_decay
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            # 不需要weight_decay
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
             },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    # 分成四组
    else:
        logger.info("use the diff learning rate")
        # the formal is basic_bert part, not include the pooler
        optimizer_grouped_parameters = [
            # BERT embeddings 和 encoder 中需要进行权重衰减的参数
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": config.weight_decay,
                "lr": config.learning_rate
            },
            # BERT embeddings 和 encoder 中不需要进行权重衰减的参数
            {
                "params": [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": config.learning_rate
             },
            # 其他部分需要/不需要权重衰减的组
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": config.weight_decay,
                "lr": config.head_learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                "weight_decay": 0.0,
                "lr": config.head_learning_rate
             },
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

    t_total = len(train_iter) * config.num_train_epochs
    # 配置线性学习率衰减策略
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=t_total * config.warmup_proportion, num_training_steps=t_total
    )

    # 打印训练信息
    logger.info("***** Running training *****")
    logger.info("  Train Num examples = %d", config.train_num_examples)
    logger.info("  Dev Num examples = %d", config.dev_num_examples)
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Instantaneous batch size GPU/CPU = %d", config.batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Train device:%s", config.device)

    model_name = config.models_name
    global_batch = 0  # 记录进行到多少batch
    dev_best_acc = 0
    train_dev_acc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    predict_all = []
    labels_all = []
    best_model = copy.deepcopy(model)

    for epoch in range(config.num_train_epochs):

        # scheduler.step() # 学习率衰减
        for epoch_batch, (input_ids, attention_mask, token_type_ids, labels) in enumerate(train_iter):
            global_batch += 1
            model.train()

            # 从 PyTorch 0.4.0 开始,PyTorch 就将 Variable 和 Tensor 合并了
            # 每个token的编号 [batch_size, sequence_length]
            input_ids = Variable(input_ids).to(config.device)
            # 形状与input_ids相同。它指示了input_ids中哪些位置是真正的tokens,哪些位置是padding的
            attention_mask = Variable(attention_mask).to(config.device)
            # 对于单句任务,所有位置都填0。对于句子对任务,第一个句子的token位置填0,第二个句子的token位置填1
            # 帮助transformer区分句子对中的两个句子
            token_type_ids = Variable(token_type_ids).to(config.device)
            # 一维的tensor,长度等于batch的大小
            labels_tensor = Variable(labels).to(config.device)

            # input_ids = input_ids.to(config.device)
            # attention_mask = attention_mask.to(config.device)
            # token_type_ids = token_type_ids.to(config.device)
            # labels_tensor = labels.to(config.device)

            outputs, loss = model(input_ids, attention_mask, token_type_ids, labels_tensor)
            # 1.模型的梯度清零。
            # 在每次反向传播之前,我们需要先清除掉之前积累的梯度,以免影响当前步的梯度计算。
            # 这是因为PyTorch中backward()函数在计算反向传播梯度时,会将新计算出的梯度值累加到已有的梯度值上。
            model.zero_grad()
            # 2.计算损失函数(loss)相对于模型里可训练参数的梯度
            # 调用loss.backward()就会计算loss对设置了requires_grad=True的所有tensor的梯度。
            # 这些梯度将被累积到每个tensor的.grad属性里。
            loss.backward()
            # 3.优化器参数更新
            # 根据网络反向传播的梯度信息来更新网络的参数,以起到降低loss函数计算值的作用。
            optimizer.step()
            # 4.调整学习率,常常跟在optimizer.step()后面使用。
            # 合适的学习率能够帮助模型更快更好地收敛。
            scheduler.step()
            predic = torch.max(outputs.data, 1)[1].cpu()
            labels_all.extend(labels)
            predict_all.extend(predic)

            if (epoch_batch + 1) % 20 == 0:
                train_acc = metrics.accuracy_score(labels_all, predict_all)
                predict_all = []
                labels_all = []
                # dev 数据
                improve = ''
                if dev_iter is not None :
                    dev_acc, dev_loss= model_evaluate(config, model, dev_iter)
                    # 如果验证集准确率优于之前的最佳准确率,且训练准确率高于 85%,则更新最佳准确率,
                    # 设置 improve 标志,并保存当前模型为最佳模型
                    if dev_acc > dev_best_acc and train_acc > 0.85:
                        dev_best_acc = dev_acc
                        improve = '*'
                        model_save(config, model, name='best_'+model_name)

                    # elif train_acc>0.90 and (train_acc+dev_acc) > train_dev_acc:
                    #     train_dev_acc=train_acc+dev_acc
                    #     model_save(config, model, name='temp_'+model_name)
                    #     improve = improve+'!'
                    else:
                        improve = ''

                time_dif = time.time() - start_time
                msg = 'Iter: {0:>4}/{1:>4},  epoch: {2:>4}/{3:>4},  Train Loss: {4:>5.6f},  Train Acc: {5:>6.2%},  Val Loss: {6:>5.6f},  Val Acc: {7:>6.2%},  Time: {8} {9}'
                logger.info(msg.format(epoch_batch, len(train_iter), epoch+1, config.num_train_epochs, loss.cpu().data.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                print(msg.format(epoch_batch, len(train_iter), epoch+1, config.num_train_epochs, loss.cpu().data.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

"""
将模型设为评估模式。
在验证集上遍历一遍,计算总损失和准确率。
过程与训练类似,但使用 torch.no_grad() 避免计算梯度,因为我们不需要更新模型参数。
返回准确率和平均损失。
"""
def model_evaluate(config, model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(data_iter):

            input_ids = Variable(input_ids).to(config.device)
            attention_mask = Variable(attention_mask).to(config.device)
            token_type_ids = Variable(token_type_ids).to(config.device)
            labels_tensor = Variable(labels).to(config.device)

            outputs, loss = model(input_ids, attention_mask, token_type_ids, labels_tensor)
            predic = torch.max(outputs.data, 1)[1].cpu()
            predict_all.extend(predic)
            labels_all.extend(labels)
            loss_total += loss.item()
        dev_acc = metrics.accuracy_score(labels_all, predict_all)
    return dev_acc, loss_total / len(data_iter),


"""
检查保存路径是否存在,不存在则创建。
构造保存的文件名,如果提供了 name 参数,则使用 name,否则使用配置中的默认文件名。
使用 torch.save 保存模型的状态字典。
"""
def model_save(config, model, name=None):
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    if name is not None:
        file_name = os.path.join(config.save_path, name + '.pkl')
    else:
        file_name = os.path.join(config.save_path, config.save_file+'.pkl')
    torch.save(model.state_dict(), file_name)
    logger.info("model saved, path: %s", file_name)

"""
构造要加载的文件名。
使用 torch.load 加载状态字典,如果是在 CPU 上加载,使用 map_location 参数
"""
def model_load(config, model, device='cpu'):
    file_name = os.path.join(config.save_path, config.save_file+'.pkl')
    logger.info('loading model: %s', file_name)
    model.load_state_dict(torch.load(file_name,
                                     map_location=device if device == 'cpu' else "{}:{}".format(device, 0)))

