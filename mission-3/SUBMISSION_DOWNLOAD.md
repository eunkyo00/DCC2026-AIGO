# Mission3 제출 파일

[GitHub 릴리스](https://github.com/eunkyo00/DCC2026-AIGO/releases/tag/mission3-submission-20261014)

최종 파일명은 `Mission3.zip`, 압축 내부 폴더는 `Mission3/`입니다. 총 2,694,292,354바이트이므로 GitHub 릴리스의 파일당 2GiB 제한에 맞춰 두 조각으로 배포합니다. **두 파일 모두 받아 합쳐야 합니다.** GitHub의 자동 생성 Source code ZIP은 모델 제출 패키지가 아닙니다.

- [Mission3.zip.001](https://github.com/eunkyo00/DCC2026-AIGO/releases/download/mission3-submission-20261014/Mission3.zip.001)
- [Mission3.zip.002](https://github.com/eunkyo00/DCC2026-AIGO/releases/download/mission3-submission-20261014/Mission3.zip.002)

## 복원

macOS/Linux에서 두 파일을 같은 폴더에 놓고 실행합니다.

```sh
cat Mission3.zip.001 Mission3.zip.002 > Mission3.zip
shasum -a 256 Mission3.zip
unzip Mission3.zip
```

Windows 명령 프롬프트(cmd):

```bat
copy /b Mission3.zip.001+Mission3.zip.002 Mission3.zip
certutil -hashfile Mission3.zip SHA256
```

SHA256:

```text
860e28c1c17ad00158d15387ab6aaf9d8cf7699d3074af0492636b4c5ce4c09c  Mission3.zip.001
2c71ee4e620ebfad87c0c2a61beb65a85018434f7a06b564c8c9a32329c3f843  Mission3.zip.002
1bfc757afd45652ad60ec7167d7cff881b10c2fb5166b2e9c6f56677e6803a6a  Mission3.zip
```

추론 코드, 모델 코드, requirements, 모델 가중치/config/tokenizer, TF-IDF 객체, 앙상블 설정, 실제 학습 로그와 해당 로그를 첨부한 학습 노트북, 실행 안내를 포함합니다. 학습 로그 첨부는 원본 JSON에서 가져왔으며 새 학습 실행 결과를 생성한 것이 아닙니다.

사용자 요청에 따라 PPT는 이 ZIP에서 제외했습니다. **대회 필수 결과 정리 PPT는 팀 제출 시 별도로 제출해야 합니다.**

```sh
cd Mission3
python -m pip install -r requirements.txt
python inference.py --audio_dir ./audio --label_dir ./json --ckpt_path ./m3_final/ensemble.json --output ./outputs/mission3.csv
```
