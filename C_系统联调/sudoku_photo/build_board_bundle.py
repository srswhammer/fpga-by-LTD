"""生成适配板上 Python 3.8 / Pillow 7 的上传包，不修改团队模型文件。"""
from pathlib import Path
import argparse
import re
import zipfile

HERE = Path(__file__).resolve().parent
MODEL = HERE.parents[1] / 'B_模型交接_v2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='sudoku_board', help='Bundle folder and ZIP name')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', args.name):
        parser.error('Use letters, digits and underscores for the bundle name.')
    prefix = args.name + '/'
    output = HERE / 'results' / (args.name + '.zip')
    output.parent.mkdir(exist_ok=True)
    source = (MODEL / 'preprocess.py').read_text(encoding='utf-8-sig')
    old = 'ImageSource = str | Path | Image.Image | np.ndarray'
    compatible = 'ImageSource = Union[str, Path, Image.Image, np.ndarray]'
    if old not in source and compatible not in source:
        raise RuntimeError('Model preprocessing changed; review compatibility before packaging.')
    if old in source:
        source = source.replace(old, compatible)
        source = source.replace('from typing import Literal', 'from typing import Literal, Union')
    source = source.replace('Image.Resampling.BICUBIC', "getattr(Image, 'Resampling', Image).BICUBIC")
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.write(HERE / 'recognize.py', prefix + 'recognize.py')
        archive.writestr(prefix + 'model/preprocess.py', source)
        for name in ('predict.py', 'self_test.npz', 'weights/int8.npz', 'weights/int8.json'):
            archive.write(MODEL / name, prefix + 'model/' + name)
    print(output)


if __name__ == '__main__':
    main()
