# 전체 29,200통화 재실험 개발본

`run_full29200.py`는 Training 전체 검사, 특징 추출, OOF 선택, 내부 평가, 전체 Training LR 재학습과 내보내기를 지원하는 단일 스크립트 개발본이다. `tests/test_full29200.py`는 데이터 완전성·분할·체크포인트·배치 처리·내보내기를 합성 데이터로 검사한다.

**실제 최종 실행·제출 확정본은 [../final_submission/](../final_submission/README.md)이다.** 이 폴더의 이전 개발 스크립트와 최종 제출 스크립트를 섞지 않는다. 최종 50:50과 공식 Validation 결과는 [상위 README](../README.md)를 따른다.

```bash
python run_full29200.py --source TRAINING_FOLDER --run-dir NEW_RESULT_FOLDER
python -m unittest discover -s tests -p 'test_*.py' -v
```

실행 패키지와 GPU 조건은 스크립트 상단에 있다. 테스트는 torch, transformers, NumPy, SciPy, scikit-learn, soundfile 등이 있는 환경을 요구한다. 원본 음성·체크포인트·특징 캐시는 포함하지 않는다.
