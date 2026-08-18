# LG Aimers 투구 제구 데이터 EDA

대용량 `train.csv`와 `trackman_history.csv`를 청크 단위로 읽어 데이터 품질,
타깃 분포, 상황별 성공률, 수치형 상관, cold-start, 선수별 이질성, Trackman
분포를 점검합니다.

```powershell
python .\eda\run_eda.py
```

기본 실행 위치는 `open` 폴더이며, 결과는 `eda/outputs` 아래에 저장됩니다.
시각화 샘플만 표본 추출을 사용하고 CSV 통계표는 전체 데이터를 집계합니다.
`trackman_joinability.json`은 메인 데이터와 Trackman 로그의 ID 네임스페이스가
직접 연결되는지도 별도로 확인합니다.

평가 데이터 5행은 형식 확인용이므로 분포 비교나 사후 통계 생성에 사용하지
않습니다. 이 원칙은 대회 규칙의 "평가 데이터 각 행 독립 예측" 제약을 반영합니다.
