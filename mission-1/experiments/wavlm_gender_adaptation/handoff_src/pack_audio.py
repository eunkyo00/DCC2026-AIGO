"""Optional sender-side packer. Packages the 27985 required WAVs only, preserving Training/ paths."""
import argparse
import zipfile
from pathlib import Path
from common import bundle, require, sha, atomic


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args()
    require(a.data_root.strip() and a.output.strip(),'Fill both absolute paths; blanks are forbidden')
    root=Path(a.data_root).expanduser().resolve();out=Path(a.output).expanduser()
    require(Path(a.data_root).expanduser().is_absolute() and out.is_absolute(),'Absolute paths required')
    require(root.is_dir() and out.parent.is_dir(),'Original data root and output parent must exist')
    require(not out.exists(),'Do not overwrite an existing archive')
    _,jobs=bundle()
    missing=[j['wav_relative_path'] for j in jobs if not (root/j['wav_relative_path']).is_file()]
    require(not missing,f'Missing {len(missing)} original WAVs; example: {missing[:3]}')
    # ZIP_STORED avoids CPU compression cost. 32GB source + archive require disk headroom.
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for i,j in enumerate(jobs):
            z.write(root/j['wav_relative_path'],j['wav_relative_path'])
            if (i+1)%1000==0:print(f'{i+1}/{len(jobs)}',flush=True)
    h=sha(out)
    atomic(out.with_name(out.name+'.sha256'),(h+'  '+out.name+'\n').encode())
    print('Send:',out,'and',out.with_name(out.name+'.sha256'))


if __name__=='__main__':main()
