"""Independent scalar arithmetic and encoding checks for team handoff."""
from pathlib import Path
import json, sys
import numpy as np
from integer_inference import IntegerDigitCNN
from preprocess import preprocess_digit_image
from verify_release import read_mem

ROOT = Path(__file__).resolve().parent

def scalar_forward(image, model):
    # Python int operations have arbitrary precision, independent of NumPy MACs.
    acc = np.empty((4,26,26), dtype=np.int32)
    relu = np.empty((4,26,26), dtype=np.uint8)
    for ch in range(4):
        for row in range(26):
            for col in range(26):
                value = int(model.conv_bias[ch])
                for kr in range(3):
                    for kc in range(3):
                        value += int(image[row+kr,col+kc])*int(model.conv_weight[ch,0,kr,kc])
                assert -(1<<31) <= value < (1<<31)
                acc[ch,row,col] = value
                relu[ch,row,col] = min(255, (max(0,value)+128)//256)
    pooled=[]
    for ch in range(4):
        for row in range(13):
            for col in range(13):
                pooled.append(max(int(relu[ch,row*2+dr,col*2+dc]) for dr in range(2) for dc in range(2)))
    scores=[]
    for digit in range(10):
        value=int(model.fc_bias[digit])
        for index in range(676):
            value += pooled[index]*int(model.fc_weight[digit,index])
        assert -(1<<31) <= value < (1<<31)
        scores.append(value)
    return acc,relu,np.array(pooled,dtype=np.uint8).reshape(4,13,13),np.array(scores,dtype=np.int32)

def main():
    model=IntegerDigitCNN.from_directory()
    fixed=np.load(ROOT/'release/reference/fixed_inputs_uint8.npy',allow_pickle=False)
    rng=np.random.default_rng(2026)
    images=np.concatenate([fixed,np.zeros((1,28,28),dtype=np.uint8),np.full((1,28,28),255,dtype=np.uint8),rng.integers(0,256,(2,28,28),dtype=np.uint8)])
    fast=model.forward(images,return_layers=True)
    for index,image in enumerate(images):
        acc,relu,pool,scores=scalar_forward(image,model)
        for expected,actual in [(acc,fast.conv_acc_int32[index]),(relu,fast.relu_uint8[index]),(pool,fast.pool_uint8[index]),(scores,fast.scores_int32[index])]:
            assert np.array_equal(expected,actual), index
    layout=json.loads((ROOT/'release/parameters/binary_layout.json').read_text())
    raw=(ROOT/'release/parameters/model_parameters_v1.bin').read_bytes()
    bin_params={}
    mem_params={}
    for entry in layout['segments']:
        name=entry['name']
        dtype=np.int8 if name.endswith('int8') else np.dtype('<i4')
        bin_params[name]=np.frombuffer(raw[entry['offset']:entry['offset']+entry['bytes']],dtype=dtype).reshape(entry['shape'])
        mem_params[name]=read_mem(ROOT/'release/parameters'/f'{name}.mem',dtype).reshape(entry['shape'])
    for parameters in [bin_params,mem_params]:
        decoded=IntegerDigitCNN(model.config,parameters).forward(images,return_layers=True)
        for field in ['conv_acc_int32','relu_uint8','pool_uint8','flatten_uint8','scores_int32']:
            assert np.array_equal(getattr(decoded,field),getattr(fast,field))
    for p in sorted((ROOT/'examples').glob('*.png')):
        prepared=preprocess_digit_image(p)
        model.forward(prepared.normalized_uint8)
    assert 'torch' not in sys.modules
    report=dict(scalar_independent_samples=len(images),scalar_layers_equal=True,
        binary_and_mem_redecoded_layers_equal=True,example_images=12,torch_imported=False,
        verified_python=sys.version.split()[0],verified_numpy=np.__version__,fpga_verified=False)
    (ROOT/'packaging_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
