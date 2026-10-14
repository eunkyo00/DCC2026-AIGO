"""Mission 1 제출용 추론 진입점.

사용법 (명령어 1회로 결과 CSV 생성):
    python inference.py --audio_dir {wav folder} --label_dir {json folder} \
        --ckpt_path {checkpoint file} --output ./outputs/mission1.csv

- 입력: WAV 폴더, 같은 파일명의 JSON 폴더. JSON에서는 발화 구간(startAt/endAt)과 speaker만 사용한다.
  신고자(speaker=1) 구간만 잘라 모델에 넣으며, gender·text·address 같은 필드는 모델 입력에 쓰지 않는다.
- 출력: CSV 한 개. 열은 [audio file name], [gender]  (gender 값은 F 또는 M)
- 모델: ckpt_path의 model.pt 하나로 충분하다 (WavLM-Large, Wav2Vec2-base 가중치와 로지스틱 회귀가 모두 들어 있어
  인터넷 연결이나 별도 다운로드가 필요 없다). 실제 추론 로직은 run_full29200.py의 predict_folder를 사용한다.
"""
import argparse
import sys
from pathlib import Path

import run_full29200 as core  # 같은 폴더의 학습·모델 코드


def main(argv=None):
    parser = argparse.ArgumentParser(description="Mission 1 gender inference (WavLM + Wav2Vec2 ensemble)")
    parser.add_argument("--audio_dir", type=Path, required=True, help="WAV 폴더")
    parser.add_argument("--label_dir", type=Path, required=True, help="JSON 폴더")
    parser.add_argument("--ckpt_path", type=Path, required=True, help="학습 완료 모델 가중치 (model.pt)")
    parser.add_argument("--output", type=Path, default=Path("./outputs/mission1.csv"), help="결과 CSV 경로")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args(argv)
    try:
        core.load_dependencies()  # torch·transformers 등 지연 로딩
        result = core.predict_folder(args.audio_dir, args.label_dir, args.ckpt_path, args.output, device=args.device)
    except Exception as error:  # 오류 원인을 한 줄로 알리고 비정상 종료 코드를 반환
        print(type(error).__name__ + ": " + str(error), file=sys.stderr)
        return 1
    print("예측 CSV: " + str(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
