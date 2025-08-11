HQD-EM: Robust VQA with Hierarchical Question Decomposition & Ensemble Margin
TL;DR: HQD-EM은 질문을 계층적으로 분해(HQD)해 편향을 낮추고, 앙상블 + 적응형 마진 로스로 일반화를 끌어올린 VQA 모델입니다.

Best model: Download [here] (https://drive.google.com/drive/folders/1-aADgu93SDutxhjZQpT5ARD_a6msxSNS?usp=drive_link)

Model Architecture
<p align="center"> <img src="assets/hqd_em_architecture.pdf" alt="HQD-EM Architecture" width="850"> </p>
Key ideas

HQD (Hierarchical Question Decomposition): 복합 질문을 (주체 → 속성 → 관계) 등 하위 질의로 분해해 언어 편향을 완화

EM (Ensemble + Margin): 서로 다른 백본/학습 seed를 앙상블하고, Adaptive Angular Margin으로 결정경계를 확장
