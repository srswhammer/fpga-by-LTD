"""Rebuild the download archive from reviewed files in models/digitcnn_v1_int8."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'models/digitcnn_v1_int8'
DOWNLOADS=ROOT/'downloads'
SKIP={'__pycache__','.ipynb_checkpoints','.venv','work','data'}

def main():
    DOWNLOADS.mkdir(exist_ok=True)
    files=sorted(p for p in SOURCE.rglob('*') if p.is_file()
                 and not SKIP.intersection(p.relative_to(SOURCE).parts)
                 and p.name!='PACKAGE_MANIFEST.json' and p.suffix!='.pyc')
    manifest={'package':'digitcnn_v1_int8_handoff_v1','files':{
        p.relative_to(SOURCE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    m=SOURCE/'PACKAGE_MANIFEST.json'
    m.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive=DOWNLOADS/'digitcnn_v1_int8_handoff_v1.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files+[m]:
            info=zipfile.ZipInfo('digitcnn_v1_int8/'+p.relative_to(SOURCE).as_posix(),date_time=(2026,9,10,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
    checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
    (DOWNLOADS/'SHA256SUMS.txt').write_text(f'{checksum}  {archive.name}\n',encoding='ascii')
    print('Archive:',archive.name,'files:',len(files)+1,'bytes:',archive.stat().st_size)
    print('SHA256:',checksum)

if __name__=='__main__': main()
