# Mission 3 최종 검수

2026-10-10, 고정 2-5 재학습 완료 모델 기준.

| 항목 | 결과 |
|---|---|
| 모델·결과 백업 복원 | Base 3개, Large, TF-IDF 및 결과 복원 |
| 새 환경 | 별도 Python 3.13 가상환경, torch 2.11.0, requirements 버전 일치 |
| 오프라인 단일 명령 | inference.py의 4개 필수 인자로 실행 성공 |
| 실제 데이터 예측 | 9개 증상을 포함하도록 선택한 Validation 7건, 정답 annotation 제거 후 기존 GPU 예측 목록과 전부 일치 |
| 경계 입력 | 빈 전사, 512 토큰을 넘는 긴 전사 각 1건 처리 성공 |
| 전체 저장 예측 재평가 | 내부 5,792건 0.6496860658420537, 공식 3,640건 0.6476169916432544 |
| CSV | file_name,symptom, JSON 문자열 목록, UTF-8 BOM, 파일명 유일성 및 증상 범위 검사 |
| 설치 의존성 | pip check 통과, requirements.txt에 torch 명시 |
| 제출 구성 | 가중치, 모델 코드, 추론 코드, requirements, 실행 출력 포함 ipynb, 보고서와 PPT |

원본 출제 PDF 11–12쪽과 대조했다. PDF는 `[label file name], [symptom]`을
지정하지만 정확한 헤더 문자열과 목록 직렬화까지 규정하지 않는다. 따라서
`file_name,symptom` 및 JSON 문자열 목록을 실행 설명에 명시했다.
별도 주최 측 채점기는 제공되지 않았다. 이 검수는 서버 제출 수락이나 비공개 Test 평가가 아니다.

전체 Validation을 새 CPU 환경에서 다시 추론한 것은 아니다. 새 환경에서는 7개 실제 표본과
2개 경계 입력으로 모든 구성 모델의 복원을 검증했고, 전체 저장 확률·CSV는 별도로 검사했다.
공식 Validation 결과로 모델·가중치·임계값을 추가 튜닝하지 않았다.

로컬 최종 전달 파일은 `M3_FINAL_SUBMISSION_2026-10-10.zip`이다.
패키지 안의 VERIFICATION.json, FINAL_AUDIT.json, SHA256SUMS.json이 상세 증거다.
모델 가중치, 원본 대화 데이터와 개별 예측은 Git에 게시하지 않는다.
