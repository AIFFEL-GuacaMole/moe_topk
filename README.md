
## MoE 

moe_topk 프로젝트는 Mixture of Experts (MoE) 모델을 구현하는 프로젝트로, Top-K gating을 활용하여 여러 Expert 모델을 효과적으로 조합하는 방식으로 동작합니다. 이 프로젝트의 구조는 모델 학습 및 평가를 체계적으로 수행할 수 있도록 설계되었습니다.



## 프로젝트 구조

```plaintext
moe_topk/
│── configs/           # 설정 파일 (하이퍼파라미터, 모델 설정 등)
│── data/              # 데이터셋 관련 코드 및 파일
│   │── preprocessing/ # 데이터 전처리 관련 코드
│   │── datasets.py    # 데이터셋 로드 및 변환 코드
│── models/            # 모델 관련 코드
│   │── moe.py         # MoE 모델 구현 (Top-K gating 포함)
│── utils/
|   | - loss.py        # loss 구현 코드 
|   |  - utils.py       # sheduler , ealry stopping     
│── README.md          # 프로젝트 설명 문서
│── MoE2.py            #  main 함수 
```


