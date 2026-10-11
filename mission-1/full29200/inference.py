"""Offline full29200 checkpoint inference; outputs only filename and predicted gender."""
import argparse
import csv
import json
from pathlib import Path


def predict_folder(audio_dir,label_dir,checkpoint,output,block_calls=32,device="auto"):
    from core import GenderEnsemble, intervals, read_segments, require
    audio_dir,label_dir,output=Path(audio_dir),Path(label_dir),Path(output)
    def index(root,suffix):
        result={}
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix.lower()==suffix and not p.name.startswith('._'):
                require(p.stem not in result,'Duplicate input stem')
                result[p.stem]=p
        return result
    wavs,labels=index(audio_dir,'.wav'),index(label_dir,'.json')
    require(bool(wavs) and set(wavs)<=set(labels),'Missing WAV or corresponding JSON')
    model=GenderEnsemble(checkpoint,device=device)
    names=sorted(wavs)
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_name(output.name+'.tmp')
    try:
        with temp.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['audio file name','gender'])
            for offset in range(0,len(names),block_calls):
                batch=names[offset:offset+block_calls]
                prepared=[]
                for stem in batch:
                    doc=json.loads(labels[stem].read_text(encoding='utf-8-sig'))
                    # Only allowed segmentation fields reach the model. Never gender, text or address.
                    prepared.append(read_segments(wavs[stem],intervals(doc)))
                p=model.predict_segments(prepared)
                for stem,pred in zip(batch,p.argmax(1)):
                    writer.writerow([wavs[stem].name,('F','M')[int(pred)]])
                stream.flush()
                print(f'Predictions {min(offset+len(batch),len(names))}/{len(names)}',flush=True)
        temp.replace(output)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audio_dir','label_dir','ckpt_path','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    a=p.parse_args()
    predict_folder(a.audio_dir,a.label_dir,a.ckpt_path,a.output,device=a.device)
