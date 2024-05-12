import json
import torch
import numpy as np
from bert import Bert
from transformers import BertTokenizer
from torch.autograd import Variable
from utils import convert_examples_to_features, BuildDataSet, read_csv
from torch.utils.data import DataLoader
from run_ernie import ErnieConfig
from run_large_roberta_pair import RobertaPairConfig
from run_large_roberta_wwm_ext import RobertaLargeConfig

def np_softmax(x):
    """
    x: (batch_size, num_classes)
    numpy版本的softmax函数
    将模型的输出转换为概率
    """
    x_row_max = x.max(axis=-1)# 找出每一行的最大值，沿着列进行操作
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])# 将x_row_max的形状变为(batch_size, 1),以便后面的广播操作
    # 当我们将exp(x - max(x))除以sum(exp(x - max(x)))时,
    # 分子和分母中的exp(-max(x))项就会消掉,得到的结果与直接计算exp(x) / sum(exp(x))是一样的。
    x = x - x_row_max# 从每一行中减去其最大值。这一步不改变softmax的结果,但可以防止后面的exp操作导致数值溢出
    x_exp = np.exp(x)
    # 调用x.reshape([64, 1])时,numpy会理解我们想要将x的形状变为(64, 1),即将其变为一个64行1列的二维数组。
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])# 算每一行exp值的和,并将其形状变为(batch_size, 1)
    softmax = x_exp / x_exp_row_sum# 每个exp值除以其所在行的exp值之和
    return softmax

def get_outputs(config, path):
    """用指定的模型和路径获取预测结果"""
    model = Bert(config).cuda()
    model.load_state_dict(torch.load(path))
    tokenizer = BertTokenizer.from_pretrained(config.tokenizer_file)
    test_examples = read_csv(config.test_path)
    test_features = convert_examples_to_features(examples=test_examples, tokenizer=tokenizer,
                                                  max_length=config.pad_size, data_type='test')
    test_dataset = BuildDataSet(test_features)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    outputs = []
    model.eval()
    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(test_loader):
            input_ids = Variable(input_ids).cuda()
            attention_mask = Variable(attention_mask).cuda()
            token_type_ids = Variable(token_type_ids).cuda()
            output, _ = model(input_ids, attention_mask, token_type_ids)
            output = output.data.cpu().numpy()
            outputs.extend(output)
        outputs = np.array(outputs)
    return np_softmax(outputs)


path_pair = '../my_model/best_roberta_large_pair.pkl'
path_wwm = '../my_model/best_roberta_wwm_large.pkl'
path_ernie = '../my_model/best_ernie.pkl'
# 设置各个模型的权重
weight1, weight2, weight3 = 0.5, 0.35, 0.05   # 8672
# 获取各个模型的预测结果
outputs_ernie = get_outputs(ErnieConfig(), path_ernie)
outputs_pair = get_outputs(RobertaPairConfig(), path_pair)
outputs_wwm = get_outputs(RobertaLargeConfig(), path_wwm)
# 按照权重融合预测结果
final_output = weight1*outputs_pair+weight2*outputs_wwm+weight3*outputs_ernie   # 设置不同模型的权重根据单个模型正确率
max = np.max(final_output, axis=1).reshape(-1, 1)
labels = np.where(final_output == max)[1]

# 写json文件，本示例代码从测试集KUAKE-QQR_test.json读取数据数据，将预测后的数据写入到KUAKE-QQR_test_pred.json：
with open('../data/KUAKE/KUAKE-QQR_test.json', 'r', encoding='UTF-8') as input_data, \
        open('../prediction_result/KUAKE-QQR_test_pred_mix3.json', 'w', encoding='UTF-8') as output_data:
    json_content = json.load(input_data)
    # 逐条读取记录，并将预测好的label赋值
    for i, block in enumerate(json_content):
        block['label'] = str(labels[i])
        # 写json文件
    json.dump(json_content, output_data, indent=2, ensure_ascii=False)
