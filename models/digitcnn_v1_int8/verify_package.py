"""Check the entire handoff manifest; run before executing model scripts."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parent

def main():
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        path=ROOT/name
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, name
    print('Package SHA-256 checks passed:',len(manifest['files']),'files')

if __name__=='__main__': main()
