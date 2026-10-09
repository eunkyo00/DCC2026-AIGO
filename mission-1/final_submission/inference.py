"""Mission 1 제출 진입점: WAV/JSON 폴더에서 통화별 성별 CSV를 생성한다."""
import argparse
import csv
import json
import math
from pathlib import Path


def caller_intervals(document):
    """허용된 startAt/endAt/speaker만 읽는다. 시간 단위는 원본의 ms이다."""
    intervals = []
    for utterance in document['utterances']:
        speaker = utterance.get('speaker', utterance.get('Speaker'))
        if speaker not in (0, 1, '0', '1'):
            raise ValueError('speaker must be 0 or 1')
        if str(speaker) != '1':
            continue
        start, end = utterance['startAt'], utterance['endAt']
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (start, end)):
            raise ValueError('Invalid interval time')
        if not 0 <= start < end:
            raise ValueError('Invalid caller interval')
        intervals.append((start / 1000, end / 1000))
    if not intervals:
        raise ValueError('No caller interval; cannot silently invent a prediction')
    return intervals


def input_pairs(audio_dir, label_dir):
    """파일 stem으로 매칭하되, 중복 이름·누락은 잘못된 매칭 대신 오류로 보고한다."""
    labels = {}
    for path in sorted(label_dir.rglob('*')):
        if path.is_file() and path.suffix.lower() == '.json':
            if path.stem in labels:
                raise ValueError(f'Duplicate JSON stem: {path.stem}')
            labels[path.stem] = path
    audio = sorted(p for p in audio_dir.rglob('*') if p.is_file() and p.suffix.lower() == '.wav')
    if not audio:
        raise ValueError('No WAV input found')
    if len({p.name for p in audio}) != len(audio):
        raise ValueError('Duplicate audio filenames')
    for path in audio:
        if path.stem not in labels:
            raise ValueError(f'Missing JSON: {path.name}')
    return [(path, labels[path.stem]) for path in audio]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audio_dir', 'label_dir', 'ckpt_path', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    pairs = input_pairs(args.audio_dir, args.label_dir)
    # 모델 의존성은 실행 시 로드해 --help에는 GPU나 가중치가 필요하지 않다.
    from model import GenderEnsemble
    model = GenderEnsemble(args.ckpt_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['audio file name', 'gender'])
            for audio, label in pairs:
                document = json.loads(label.read_text(encoding='utf-8-sig'))
                # gender/text 등 비허용 라벨은 추론 모델에 전달하지 않는다.
                prediction = model.predict(audio, caller_intervals(document))
                writer.writerow([audio.name, prediction])
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f'Saved {len(pairs)} predictions: {args.output}')


if __name__ == '__main__':
    main()
