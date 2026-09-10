"""Run the packaged integer reference with no dataset download or PyTorch."""
from pathlib import Path
import argparse
import numpy as np
from integer_inference import IntegerDigitCNN
from preprocess import preprocess_digit_image

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()
    model = IntegerDigitCNN.from_directory()
    if args.image:
        inputs = preprocess_digit_image(args.image).normalized_uint8
    else:
        inputs = np.load(ROOT/'release/reference/fixed_inputs_uint8.npy', allow_pickle=False)
    result = model.forward(inputs)
    print('Predictions:', result.predictions.tolist())
    print('Scores int32:', result.scores_int32.tolist())

if __name__ == '__main__':
    main()
