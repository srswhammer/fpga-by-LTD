"""整张照片 → 找棋盘 → 透视矫正 → 光照校正 → 切成 81 格。

来源：dyx_camera/processing/recognize.py
  - isolate_digit()  原样搬过来
  - split()          来自 process_photo() 里 cells_only=True 走的那一段，算法一行没改
改动：
  - 去掉了对 model/ 文件夹的检查和加载（切格本身用不到软件模型）
  - 输入可以直接给图像数组，返回 81 个数组，不必先存 PNG 再读回来
  - 存图改成可选：给了 save_dir / cells_dir 才保存
软件 INT8 识别那部分没有搬过来，完整原版留在 reference/ 里做对照。
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SIZE = 450          # 矫正后棋盘的边长（像素）


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


def _ordered(points):
    p=np.asarray(points,dtype=np.float32).reshape(4,2)
    sums=p.sum(axis=1); delta=np.diff(p,axis=1).ravel()
    return p[[sums.argmin(),delta.argmin(),sums.argmax(),delta.argmax()]]


def _warp(points,image,margin=0):
    end=SIZE-1-margin
    dest=np.float32([[margin,margin],[end,margin],[end,end],[margin,end]])
    matrix=cv2.getPerspectiveTransform(_ordered(points),dest)
    return cv2.warpPerspective(image,matrix,(SIZE,SIZE)),matrix


def _grid_maps(board):
    g=cv2.cvtColor(board,cv2.COLOR_RGB2GRAY)
    b=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,10)
    h=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((1,75),np.uint8))
    v=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((75,1),np.uint8))
    return h,v


def _peaks(projection):
    result=[]; strength=[]
    for expected in np.linspace(0,SIZE-1,10):
        a=max(0,int(round(expected))-9); b=min(SIZE,int(round(expected))+10)
        k=a+int(np.argmax(projection[a:b]))
        if projection[k] < .40: k=int(round(expected))
        result.append(k); strength.append(float(projection[k]))
    return result,strength


def _load_photo(photo):
    """照片可以是文件路径，也可以是 (H, W, 3) 的 RGB uint8 数组。"""
    if isinstance(photo, (str, Path)):
        return np.array(Image.open(str(photo)).convert('RGB'))
    photo = np.asarray(photo, dtype=np.uint8)
    if photo.ndim != 3 or photo.shape[2] != 3:
        raise ValueError('photo 必须是 RGB 图像 (H, W, 3)，实际是 %s' % (photo.shape,))
    return photo


def split(photo, save_dir=None, cells_dir=None):
    """切出 81 格，返回 81 个灰度 uint8 数组（从左到右、从上到下）。

    save_dir：给了就保存检查用的中间图（01/02/02b 和 cells_raw/）
    cells_dir：给了就把 81 格存成 00.png … 80.png（和原来交给组长的格式一样）
    """
    photo = _load_photo(photo)
    gray=cv2.cvtColor(photo,cv2.COLOR_RGB2GRAY)
    binary=cv2.adaptiveThreshold(cv2.GaussianBlur(gray,(5,5),0),255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,9)

    # ---- 1. 找棋盘：在所有四边形轮廓里挑网格线最像的那个 ----
    contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not gray.size*.12<area<gray.size*.95: continue
        approx=cv2.approxPolyDP(contour,.02*cv2.arcLength(contour,True),True)
        if len(approx)!=4 or not cv2.isContourConvex(approx): continue
        corners=_ordered(approx)
        lengths=np.linalg.norm(corners-np.roll(corners,1,axis=0),axis=1)
        if max(lengths)/min(lengths)>1.65: continue
        board,matrix=_warp(corners,photo)
        h,v=_grid_maps(board)
        ys,hs=_peaks((h>0).mean(axis=1)); xs,vs=_peaks((v>0).mean(axis=0))
        major=[hs[i] for i in (0,3,6,9)]+[vs[i] for i in (0,3,6,9)]
        matches=sum(x>.45 for x in hs+vs)+3*sum(x>.60 for x in major)
        candidates.append((matches+sum(hs+vs)/20,area,corners,board,matrix,xs,ys,hs,vs))
    if not candidates: raise RuntimeError('No complete grid detected; retake with all four corners visible.')
    best=max(candidates,key=lambda c:c[0])
    score,area,corners,board,matrix,xs,ys,hs,vs=best
    if any(positions[i] < .60 for positions in (hs,vs) for i in (0,3,6,9)):
        # Contour approximation can clip an outer border during rectification.
        # Recheck with a margin, without changing the digit sampling geometry.
        padded,_=_warp(corners,photo,margin=4)
        ph,pv=_grid_maps(padded)
        _,check_h=_peaks((ph>0).mean(axis=1))
        _,check_v=_peaks((pv>0).mean(axis=0))
        if any(values[i] < .60 for values in (check_h,check_v) for i in (0,3,6,9)):
            raise RuntimeError('Grid candidate failed major-line verification.')
    xs[0]=ys[0]=0; xs[-1]=ys[-1]=SIZE-1
    assert len(xs)==len(ys)==10 and all(b>a for a,b in zip(xs,xs[1:])) and all(b>a for a,b in zip(ys,ys[1:]))

    # ---- 2. 光照校正：估计纸面亮度再除掉，减轻阴影 ----
    board_gray=cv2.medianBlur(cv2.cvtColor(board,cv2.COLOR_RGB2GRAY),3)
    background=cv2.morphologyEx(board_gray,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    corrected=cv2.divide(board_gray,np.maximum(background,1),scale=255)

    if save_dir is not None:
        save_dir=Path(save_dir); save_dir.mkdir(parents=True,exist_ok=True)
        overlay=photo.copy()
        cv2.polylines(overlay,[corners.astype(np.int32)],True,(0,255,0),2)
        for k,(x,y) in enumerate(corners):
            cv2.circle(overlay,(int(x),int(y)),4,(255,0,0),-1)
            cv2.putText(overlay,str(k+1),(int(x)+5,int(y)-5),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1)
        Image.fromarray(overlay).save(save_dir/'01_detected_board.png')
        Image.fromarray(board).save(save_dir/'02_rectified_board.png')
        Image.fromarray(corrected).save(save_dir/'02b_illumination_corrected.png')
        raw_dir=save_dir/'cells_raw'; raw_dir.mkdir(exist_ok=True)
    if cells_dir is not None:
        cells_dir=Path(cells_dir); cells_dir.mkdir(parents=True,exist_ok=True)

    # ---- 3. 切 81 格：每格四周各去掉 10%，再去掉杂点和残留网格线 ----
    cells=[]
    for row in range(9):
        for col in range(9):
            x0,x1=xs[col:col+2]; y0,y1=ys[row:row+2]
            mx=max(4,int(round((x1-x0)*.10))); my=max(4,int(round((y1-y0)*.10)))
            clean_cell=isolate_digit(corrected[y0+my:y1-my,x0+mx:x1-mx])
            cells.append(clean_cell)
            if save_dir is not None:
                Image.fromarray(board[y0+my:y1-my,x0+mx:x1-mx]).save(raw_dir/f'r{row+1}_c{col+1}.png')
            if cells_dir is not None:
                Image.fromarray(clean_cell).save(cells_dir/f'{row*9+col:02d}.png')
    return cells


def load_cells(directory):
    """读回 00.png … 80.png，返回 81 个灰度数组（分步调试用）。"""
    directory = Path(directory)
    paths = [directory / ('%02d.png' % i) for i in range(81)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('%s 里缺少 %d 张格子图，例如 %s' % (directory, len(missing), missing[0]))
    return [np.array(Image.open(str(p)).convert('L')) for p in paths]
