"""v2.4 数独单格识别：浮点预测、INT8 参考计算及十类自检。"""

# 导入库与路径；所有参数均从本文件所在目录读取
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from preprocess import preprocess_sudoku_cell

HERE = Path(__file__).resolve().parent
CLASS_NAMES = ['空白', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# 对外输出9位独热字符串：最低位代表1，最高位代表9，空白为全零
def format_prediction(label):
    label = int(label)
    return f'{0 if label == 0 else 1 << (label - 1):09b}'


# CNN 结构：两组卷积池化，再接一个全连接层，共 2346 个参数
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1×28×28 → 4×26×26 → 4×13×13
        self.conv1 = nn.Conv2d(1, 4, 3, stride=1, padding=0)
        # 4×13×13 → 8×11×11 → 8×5×5
        self.conv2 = nn.Conv2d(4, 8, 3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=False)
        self.fc = nn.Linear(8 * 5 * 5, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(x.flatten(1))


# INT8 参考运算：使用已验证的整数参数；INT64中间计算、INT32输出
def integer_scores(inputs, params, manifest):
    x = inputs[:, None]
    for name in ('conv1', 'conv2'):
        windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(2, 3))
        acc = np.einsum('nchwkl,ockl->nohw', windows.astype(np.int64),
                        params[name + '_weight'].astype(np.int64), optimize=True)
        acc += params[name + '_bias'][None, :, None, None]
        shift = manifest['layers'][name]['right_shift']
        if shift > 0:
            acc = (acc + (1 << (shift - 1))) >> shift
        elif shift < 0:
            acc = acc << (-shift)
        x = np.clip(acc, 0, 255).astype(np.uint8)
        n, c, h, w = x.shape
        x = x[:, :, :h // 2 * 2, :w // 2 * 2]
        x = x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))
    out = x.reshape(len(x), -1).astype(np.int64) @ params['fc_weight'].astype(np.int64).T
    out += params['fc_bias']
    return out.astype(np.int32)


# 加载一次，随后可反复传入单张或一批 28×28 的 UINT8 图片
class Predictor:
    def __init__(self, backend='fp32'):
        self.backend = backend
        if backend == 'fp32':
            torch.set_num_threads(4)
            checkpoint = torch.load(HERE / 'weights/fp32.pt', map_location='cpu', weights_only=True)
            self.model = DigitCNN().eval()
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            with np.load(HERE / 'weights/int8.npz', allow_pickle=False) as data:
                self.params = {key: data[key].copy() for key in data.files}
            self.manifest = json.loads((HERE / 'weights/int8.json').read_text(encoding='utf-8'))

    def scores(self, images):
        images = np.asarray(images)
        if images.ndim == 2:
            images = images[None]
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(images), 128):
                batch = images[start:start + 128]
                if self.backend == 'fp32':
                    # 与训练保持一致：除以 256，不是 255
                    x = torch.from_numpy(batch.copy()).unsqueeze(1).float() / 256
                    outputs.append(self.model(x).numpy())
                else:
                    outputs.append(integer_scores(batch, self.params, self.manifest))
        return np.concatenate(outputs)

    def predict_one_hot(self, images):
        """返回每张图片的9位独热字符串列表及十类原始分数。"""
        scores = self.scores(images)
        return [format_prediction(label) for label in scores.argmax(1)], scores

    def predict_image(self, image, polarity='auto'):
        """接收单格图片，预处理后返回9位字符串；模型参数可重复使用。"""
        result = preprocess_sudoku_cell(image, polarity=polarity)
        codes, _ = self.predict_one_hot(result.normalized_uint8)
        if 'warning' in result.metadata:
            print(result.metadata['warning'], file=sys.stderr)
        return codes[0]


# 自检用于检查文件完整性与运行环境，不等同于完整测试集准确率
def self_test():
    with np.load(HERE / 'self_test.npz', allow_pickle=False) as data:
        images, labels = data['inputs_uint8'], data['labels']
    blanks = [preprocess_sudoku_cell(np.full((64, 64), value, dtype=np.uint8)).normalized_uint8
              for value in (0, 255)]
    images = np.concatenate([images, blanks])
    labels = np.concatenate([labels, [0, 0]])
    expected_codes = [
        '000000000', '000000001', '000000010', '000000100', '000001000',
        '000010000', '000100000', '001000000', '010000000', '100000000'
    ]
    for backend in ('fp32', 'int8'):
        codes, scores = Predictor(backend).predict_one_hot(images)
        predictions = scores.argmax(1)
        if not np.array_equal(predictions, labels):
            raise RuntimeError(f'{backend} 自检失败：{predictions.tolist()}')
        if codes != [expected_codes[label] for label in labels]:
            raise RuntimeError(f'{backend} 独热编码自检失败：{codes}')
        print(f'{backend}：空白及数字 1～9、黑底/白底空白、9位独热编码自检通过。')


# 命令入口：输入应是裁好的单格，不是整张数独
def main():
    parser = argparse.ArgumentParser(description='v2.4 数独单格识别（空白及 1～9）')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--image', type=Path, help='单个数字或空白格图片')
    action.add_argument('--self-test', action='store_true', help='同时自检浮点版与整数版')
    parser.add_argument('--backend', choices=['fp32', 'int8'], default='fp32')
    parser.add_argument('--polarity', choices=['auto', 'dark_on_light', 'light_on_dark'], default='auto')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    print(Predictor(args.backend).predict_image(args.image, args.polarity))


if __name__ == '__main__':
    main()
