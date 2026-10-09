"""整张数独照片 → 81 格 → TBR INT8 识别。运行方法见 README.md。"""
from pathlib import Path
import argparse
import ast
import json
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw

DEFAULT_MODEL = Path(__file__).resolve().parent / 'model'
if not DEFAULT_MODEL.is_dir():
    DEFAULT_MODEL = Path(__file__).resolve().parents[2] / 'B_模型交接_v2'
RESAMPLING = getattr(Image, 'Resampling', Image)


def isolate_digit(cell):
    """保留主要连通笔画及一像素邻域，避免杂点扩大数字归一化范围。"""
    border = np.concatenate([cell[0], cell[-1], cell[:, 0], cell[:, -1]])
    ink = np.maximum(float(np.median(border)) - cell.astype(float), 0)
    # Preserve the model's blank-cell contrast decision before removing background.
    if float(np.percentile(ink, 99.5)) < 20:
        return cell
    threshold = max(8.0, float(np.percentile(ink, 99.5)) * 0.2)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (ink >= threshold).astype(np.uint8), 8)
    if count <= 1:
        return cell
    # A faint horizontal grid remnant can survive the geometric crop.
    # Remove only wide, shallow components attached to the top/bottom edge.
    cleaned_edges = cell.copy()
    removed = False
    height, width = cell.shape
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if w >= width * .6 and h <= height * .15 and (y == 0 or y+h == height):
            region = cv2.dilate((labels == label).astype(np.uint8), np.ones((3,3),np.uint8))
            cleaned_edges[region > 0] = int(np.median(border))
            removed = True
    if removed:
        return cleaned_edges
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[largest, cv2.CC_STAT_AREA] < 8:
        return cell
    keep = cv2.dilate((labels == largest).astype(np.uint8), np.ones((3, 3), np.uint8))
    cleaned = cell.copy()
    cleaned[keep == 0] = 255
    return cleaned


def process_photo(source, output, model=DEFAULT_MODEL):
    """识别一张照片，保存检查材料并返回结果字典；不求解、不发送串口。"""
    SOURCE = Path(source).resolve()
    ROOT = Path(output).resolve()
    MODEL = Path(model).resolve()
    if not SOURCE.is_file():
        raise FileNotFoundError(f'Photo not found: {SOURCE}')
    for name in ('preprocess.py', 'predict.py', 'weights/int8.npz', 'weights/int8.json', 'self_test.npz'):
        if not (MODEL / name).is_file():
            raise FileNotFoundError(f'Model file not found: {MODEL / name}')
    # Load the selected model preprocessing module without altering the team package.
    import importlib.util
    spec = importlib.util.spec_from_file_location('_sudoku_tbr_preprocess', MODEL / 'preprocess.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    preprocess_sudoku_cell = module.preprocess_sudoku_cell
    ROOT.mkdir(parents=True, exist_ok=True)
    SIZE=450
    photo=np.array(Image.open(SOURCE).convert('RGB'))
    gray=cv2.cvtColor(photo,cv2.COLOR_RGB2GRAY)
    binary=cv2.adaptiveThreshold(cv2.GaussianBlur(gray,(5,5),0),255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9)

    def ordered(points):
        p=np.asarray(points,dtype=np.float32).reshape(4,2)
        sums=p.sum(axis=1); delta=np.diff(p,axis=1).ravel()
        return p[[sums.argmin(),delta.argmin(),sums.argmax(),delta.argmax()]]

    def warp(points,image,margin=0):
        end=SIZE-1-margin
        dest=np.float32([[margin,margin],[end,margin],[end,end],[margin,end]])
        matrix=cv2.getPerspectiveTransform(ordered(points),dest)
        return cv2.warpPerspective(image,matrix,(SIZE,SIZE)),matrix

    def grid_maps(board):
        g=cv2.cvtColor(board,cv2.COLOR_RGB2GRAY)
        b=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,10)
        h=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((1,75),np.uint8))
        v=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((75,1),np.uint8))
        return h,v

    def peaks(projection):
        result=[]; strength=[]
        for expected in np.linspace(0,SIZE-1,10):
            a=max(0,int(round(expected))-9); b=min(SIZE,int(round(expected))+10)
            k=a+int(np.argmax(projection[a:b]))
            if projection[k] < .40: k=int(round(expected))
            result.append(k); strength.append(float(projection[k]))
        return result,strength

    contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not gray.size*.12<area<gray.size*.95: continue
        approx=cv2.approxPolyDP(contour,.02*cv2.arcLength(contour,True),True)
        if len(approx)!=4 or not cv2.isContourConvex(approx): continue
        corners=ordered(approx)
        lengths=np.linalg.norm(corners-np.roll(corners,1,axis=0),axis=1)
        if max(lengths)/min(lengths)>1.65: continue
        board,matrix=warp(corners,photo)
        h,v=grid_maps(board)
        ys,hs=peaks((h>0).mean(axis=1)); xs,vs=peaks((v>0).mean(axis=0))
        major=[hs[i] for i in (0,3,6,9)]+[vs[i] for i in (0,3,6,9)]
        matches=sum(x>.45 for x in hs+vs)+3*sum(x>.60 for x in major)
        candidates.append((matches+sum(hs+vs)/20,area,corners,board,matrix,xs,ys,hs,vs))
    if not candidates: raise RuntimeError('No complete grid detected; retake with all four corners visible.')
    best=max(candidates,key=lambda c:c[0])
    score,area,corners,board,matrix,xs,ys,hs,vs=best
    if any(positions[i] < .60 for positions in (hs,vs) for i in (0,3,6,9)):
        # Contour approximation can clip an outer border during rectification.
        # Recheck with a margin, without changing the digit sampling geometry.
        padded,_=warp(corners,photo,margin=4)
        ph,pv=grid_maps(padded)
        _,check_h=peaks((ph>0).mean(axis=1))
        _,check_v=peaks((pv>0).mean(axis=0))
        if any(values[i] < .60 for values in (check_h,check_v) for i in (0,3,6,9)):
            raise RuntimeError('Grid candidate failed major-line verification.')
    xs[0]=ys[0]=0; xs[-1]=ys[-1]=SIZE-1
    assert len(xs)==len(ys)==10 and all(b>a for a,b in zip(xs,xs[1:])) and all(b>a for a,b in zip(ys,ys[1:]))
    overlay=photo.copy()
    cv2.polylines(overlay,[corners.astype(np.int32)],True,(0,255,0),2)
    for k,(x,y) in enumerate(corners):
        cv2.circle(overlay,(int(x),int(y)),4,(255,0,0),-1)
        cv2.putText(overlay,str(k+1),(int(x)+5,int(y)-5),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1)
    Image.fromarray(overlay).save(ROOT/'01_detected_board.png')
    Image.fromarray(board).save(ROOT/'02_rectified_board.png')
    # Estimate local paper brightness and divide it out before TBR preprocessing.
    # This addresses broad shadows; no model or expected labels enter the correction.
    board_gray=cv2.medianBlur(cv2.cvtColor(board,cv2.COLOR_RGB2GRAY),3)
    background=cv2.morphologyEx(board_gray,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    corrected=cv2.divide(board_gray,np.maximum(background,1),scale=255)
    # An unfiltered comparison exposes digits whose thin strokes change under median filtering.
    # Disagreement is a review signal; never select a label using the known Sudoku answer.
    unfiltered_gray=cv2.cvtColor(board,cv2.COLOR_RGB2GRAY)
    comparison_background=cv2.morphologyEx(unfiltered_gray,cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    comparison_corrected=cv2.divide(unfiltered_gray,np.maximum(comparison_background,1),scale=255)
    Image.fromarray(corrected).save(ROOT/'02b_illumination_corrected.png')
    raw_dir=ROOT/'cells_raw'; normalized_dir=ROOT/'cells_28x28'
    raw_dir.mkdir(exist_ok=True); normalized_dir.mkdir(exist_ok=True)
    inputs=[]; comparison_inputs=[]; metadata=[]; raw_cells=[]
    for row in range(9):
        for col in range(9):
            x0,x1=xs[col:col+2]; y0,y1=ys[row:row+2]
            mx=max(4,int(round((x1-x0)*.10))); my=max(4,int(round((y1-y0)*.10)))
            cell=board[y0+my:y1-my,x0+mx:x1-mx]
            raw_cells.append(cell)
            name=f'r{row+1}_c{col+1}.png'
            Image.fromarray(cell).save(raw_dir/name)
            clean_cell=isolate_digit(corrected[y0+my:y1-my,x0+mx:x1-mx])
            result=preprocess_sudoku_cell(clean_cell,polarity='dark_on_light')
            inputs.append(result.normalized_uint8)
            comparison_cell=isolate_digit(comparison_corrected[y0+my:y1-my,x0+mx:x1-mx])
            comparison_inputs.append(preprocess_sudoku_cell(
                comparison_cell,polarity='dark_on_light').normalized_uint8)
            Image.fromarray(result.normalized_uint8).save(normalized_dir/name)
            metadata.append({'row':row+1,'column':col+1,'crop_xyxy':[x0+mx,y0+my,x1-mx,y1-my],**result.metadata})
    inputs=np.stack(inputs)
    assert inputs.shape==(81,28,28) and inputs.dtype==np.uint8
    np.save(ROOT/'model_inputs_uint8.npy',inputs)
    (ROOT/'model_inputs_uint8.bin').write_bytes(inputs.tobytes())

    def contact_sheet(images,path,scale=2):
        tile=84; canvas=Image.new('RGB',(9*tile,9*tile),(238,238,238)); draw=ImageDraw.Draw(canvas)
        for i,im in enumerate(images):
            r,c=divmod(i,9)
            pic=Image.fromarray(im).convert('RGB'); pic.thumbnail((64,64))
            if scale>1: pic=pic.resize((56,56),RESAMPLING.NEAREST)
            canvas.paste(pic,(c*tile+10,r*tile+20))
            draw.text((c*tile+10,r*tile+4),f'r{r+1} c{c+1}',fill=(20,20,20))
        canvas.save(path)
    contact_sheet(raw_cells,ROOT/'03_raw_cells_contact.png',1)
    contact_sheet(inputs,ROOT/'04_model_inputs_contact.png',2)

    # Use the exact existing NumPy INT8 inference function, without requiring Torch.
    # No changes to TBR's source, weights, shifts or preprocessing.
    tree=ast.parse((MODEL/'predict.py').read_text(encoding='utf-8-sig'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='integer_scores')
    namespace={'np':np}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'TBR_integer_scores','exec'),namespace)
    infer=namespace['integer_scores']
    with np.load(MODEL/'weights/int8.npz',allow_pickle=False) as archive:
        params={k:archive[k].copy() for k in archive.files}
    manifest=json.loads((MODEL/'weights/int8.json').read_text(encoding='utf-8'))
    with np.load(MODEL/'self_test.npz',allow_pickle=False) as test:
        test_inputs=test['inputs_uint8']; test_labels=test['labels']
    if not np.array_equal(infer(test_inputs,params,manifest).argmax(axis=1),test_labels):
        raise RuntimeError('TBR model self-test failed')
    # Small batches limit convolution workspace on the Zynq ARM processor.
    scores=np.concatenate([infer(inputs[i:i+9],params,manifest) for i in range(0,len(inputs),9)])
    digits=scores.argmax(axis=1)
    comparison_inputs=np.stack(comparison_inputs)
    comparison_scores=np.concatenate([infer(comparison_inputs[i:i+9],params,manifest)
                                      for i in range(0,len(comparison_inputs),9)])
    comparison_digits=comparison_scores.argmax(axis=1)
    review_cells=[{'row':i//9+1,'column':i%9+1,
                  'filtered_digit':int(digits[i]),'unfiltered_digit':int(comparison_digits[i])}
                 for i in range(81) if digits[i]!=comparison_digits[i]]
    np.save(ROOT/'comparison_inputs_uint8.npy',comparison_inputs)
    contact_sheet(comparison_inputs,ROOT/'05_comparison_inputs_contact.png',2)
    codes=[f'{0 if d==0 else 1<<(int(d)-1):09b}' for d in digits]
    result={'source':str(SOURCE),'source_size_wh':[photo.shape[1],photo.shape[0]],
        'corners_tl_tr_br_bl':corners.tolist(),'rectified_size_wh':[SIZE,SIZE],
        'original_board_approx_pixels':int(np.sqrt(area)),
        'warp_is_resampling_not_new_detail':True,
        'grid_x_positions':xs,'grid_y_positions':ys,
        'input_shape':[81,28,28],'input_dtype':'uint8','order':'row-major left-to-right top-to-bottom',
        'predicted_digits':digits.tolist(),'predicted_one_hot':codes,'self_test_passed':True,
        'comparison_digits':comparison_digits.tolist(),'review_cells':review_cells,
        'requires_review':bool(review_cells),
        'inference':'existing TBR NumPy INT8 function; PC software inference, not FPGA CNN',
        'capture_preprocessing':'3x3 median; 21x21 grayscale closing brightness correction; weak minor grid lines interpolated from validated major lines',
        'metadata':metadata}
    (ROOT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'predicted_board.txt').write_text('\n'.join(' '.join(map(str,digits[r*9:r*9+9])) for r in range(9))+'\n',encoding='ascii')
    np.save(ROOT/'scores_int32.npy',scores)

    (ROOT/'original_81bytes.bin').write_bytes(bytes(int(d) for d in digits))
    (ROOT/'original_81bytes.hex').write_text(' '.join(f'{int(d):02X}' for d in digits)+'\n',encoding='ascii')
    (ROOT/'one_hot_81.txt').write_text('\n'.join(codes)+'\n',encoding='ascii')
    print((ROOT/'predicted_board.txt').read_text(encoding='ascii'))
    print(f'Output: {ROOT}')
    if review_cells:
        print('REVIEW REQUIRED: filtered/unfiltered predictions disagree:')
        for item in review_cells:
            print('row {row}, column {column}: {filtered_digit} / {unfiltered_digit}'.format(**item))
        print('Check the original photo before handing this board to a solver. Agreement is not proof of correctness.')
    return result


def main():
    parser = argparse.ArgumentParser(description='Recognize a complete Sudoku photo using the team INT8 model.')
    parser.add_argument('photo', type=Path, help='Input photo; keep all four board corners visible')
    parser.add_argument('--output', type=Path, help='New or empty output directory')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='Trusted team model directory')
    args = parser.parse_args()
    output = args.output or Path(__file__).resolve().parent / 'results' / args.photo.stem
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; choose another --output directory.')
    try:
        process_photo(args.photo, output, args.model)
    except (OSError, ValueError, RuntimeError, StopIteration) as exc:
        parser.exit(1, f'Failed: {exc}\n')


if __name__ == '__main__':
    main()
