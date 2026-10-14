"""Short notebook cells call these widgets; long paths/code/logs stay out of cell output."""
import html
import json
import os
import shutil
import time
import sys
import contextlib
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ipywidgets as W
from IPython.display import display, clear_output

ROOT=Path('/content/drive/MyDrive')
OUTPUT_ROOT=ROOT/'AIGO_Mission1_full29200'
SOURCES=[]
CONFIG=None
RESULT=None


def setup():
    global CONFIG
    current=[ROOT]
    chooser=W.SelectMultiple(options=[],rows=9,layout=W.Layout(width='95%'))
    location=W.HTML()
    selected=W.HTML('선택된 원본: 0개')
    status=W.HTML()
    open_button=W.Button(description='폴더 열기')
    up=W.Button(description='상위 폴더')
    add=W.Button(description='선택 항목 추가',button_style='info')
    reset=W.Button(description='선택 초기화')
    new_name='full29200_'+datetime.now(timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    run_name=W.Text(value=new_name,description='실행 이름:',layout=W.Layout(width='95%'))
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    existing=sorted((p.name for p in OUTPUT_ROOT.iterdir() if p.is_dir()),reverse=True)
    resume=W.Dropdown(options=[('새 실험','')]+[(s,s) for s in existing],description='재시작:',layout=W.Layout(width='95%'))
    confirm=W.Button(description='설정 저장',button_style='success')

    def refresh():
        location.value='<b>현재 폴더:</b> …/'+html.escape('/'.join(current[0].parts[-3:]))
        try:
            entries=sorted(current[0].iterdir(),key=lambda p:(not p.is_dir(),p.name.lower()))
            chooser.options=[(('📁 ' if p.is_dir() else 'ZIP ')+p.name,str(p)) for p in entries
                             if p.is_dir() or p.suffix.lower()=='.zip']
        except OSError as e:
            status.value=html.escape(str(e))
    def navigate(_):
        if len(chooser.value)==1 and Path(chooser.value[0]).is_dir():
            current[0]=Path(chooser.value[0]);refresh()
    def parent(_):
        if current[0]!=ROOT:
            current[0]=current[0].parent;refresh()
    def include(_):
        for p in chooser.value:
            if p not in SOURCES:
                SOURCES.append(p)
        selected.value='선택된 원본: '+str(len(SOURCES))+'개<br>'+ '<br>'.join(html.escape(Path(p).name) for p in SOURCES)
    def empty(_):
        SOURCES.clear();selected.value='선택된 원본: 0개'
    def change(e):
        run_name.value=e['new'] or new_name
        run_name.disabled=bool(e['new'])
        if e['new']:
            p=OUTPUT_ROOT/e['new']/'run_config.json'
            if p.exists():
                SOURCES[:]=json.loads(p.read_text())['sources']
                selected.value='재시작 원본 '+str(len(SOURCES))+'개: '+', '.join(html.escape(Path(v).name) for v in SOURCES)
    def store(_):
        global CONFIG
        name=run_name.value.strip()
        if not SOURCES or not name or Path(name).name!=name or name in ('.','..'):
            status.value='Training ZIP/폴더를 추가하고 실행 이름을 확인하세요.';return
        CONFIG={'sources':list(SOURCES),'output':str(OUTPUT_ROOT/name)}
        Path('/content/aigo_runtime/config.json').write_text(json.dumps(CONFIG,ensure_ascii=False,indent=2))
        status.value='<b>설정 저장 완료.</b> 다음 실행 셀로 진행하세요. 같은 실행 이름은 중단 지점부터 이어집니다.'
    open_button.on_click(navigate);up.on_click(parent);add.on_click(include);reset.on_click(empty)
    resume.observe(change,names='value');confirm.on_click(store)
    refresh()
    display(W.VBox([W.HTML('<b>Training 데이터만 선택하세요.</b> 폴더를 열어 ZIP 또는 Training 폴더를 추가하세요. WAV/JSON ZIP이 따로 있으면 둘 다 추가합니다. 공유 폴더는 내 드라이브 바로가기를 먼저 만드세요.'),
                    location,chooser,W.HBox([up,open_button,add,reset]),selected,resume,run_name,confirm,status]))


def execute():
    global RESULT
    from pipeline import run
    if CONFIG is None:
        print('위 설정에서 원본을 선택하고 「설정 저장」을 누르세요.');return
    panel=W.Output(layout=W.Layout(border='1px solid #ddd',padding='10px'))
    display(panel)
    latest=[]
    console=sys.stdout
    def progress(message):
        latest.append(message)
        with contextlib.redirect_stdout(console), panel:
            clear_output(wait=True)
            print('\n'.join(latest[-3:]))
    # Log stdout/stderr (HF/sklearn/etc.) to a file, not tens of thousands of notebook lines.
    import contextlib
    out=Path(CONFIG['output']);out.mkdir(parents=True,exist_ok=True)
    def observed(message):
        with (out/'events.jsonl').open('a') as f:
            f.write(json.dumps({'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'message':message},ensure_ascii=False)+'\n')
        progress(message)
    try:
        with open('/content/aigo_runtime/runtime.log','a',buffering=1) as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                RESULT=run(CONFIG,observed)
        shutil.copyfile('/content/aigo_runtime/runtime.log',out/'runtime.log')
    except KeyboardInterrupt:
        progress('중단됨. 이미 Drive에 저장된 블록은 유지됩니다. 같은 실행 이름으로 실행하면 이어집니다.')
    except Exception as e:
        progress('실행 중단: '+str(e)[:600])
        progress('상세 기록: 실행 폴더의 last_failure.json. 아래 진단 ZIP 다운로드를 사용할 수 있습니다.')


def downloads():
    from google.colab import files
    if CONFIG is None:
        print('설정을 먼저 저장하세요. 재접속했다면 기존 실행을 선택하세요.');return
    out=Path(CONFIG['output'])
    button=W.Button(description='최종 ZIP 다운로드',button_style='success')
    diagnostic=W.Button(description='중간 기록 ZIP 다운로드')
    label=W.HTML()
    result=out/'Mission1_full29200_result.zip'
    label.value=('최종 ZIP: '+f'{result.stat().st_size/(1<<30):.2f}GB · Drive에도 저장됨' if result.exists()
                 else '아직 완료되지 않았습니다. 중간 기록은 다운로드할 수 있습니다.')
    def final(_):
        if not result.exists():
            label.value='최종 ZIP이 아직 없습니다. 실행 완료 후 다시 누르세요.';return
        local=Path('/content')/result.name
        if not local.exists() or local.stat().st_size!=result.stat().st_size:
            shutil.copyfile(result,local)
        files.download(str(local))
    def partial(_):
        archive=Path('/content/Mission1_intermediate_records.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in out.rglob('*'):
                if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.md','.log'):
                    z.write(p,str(p.relative_to(out)))
        files.download(str(archive))
    button.on_click(final);diagnostic.on_click(partial)
    display(W.VBox([label,W.HBox([button,diagnostic])]))
