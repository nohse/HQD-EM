HQD-EM: Robust VQA with Hierarchical Question Decomposition & Ensemble Margin
TL;DR: HQD-EM은 질문을 계층적으로 분해(HQD)해 편향을 낮추고, 앙상블 + 적응형 마진 로스로 일반화를 끌어올린 VQA 모델입니다.

Best model: Download [here] (https://drive.google.com/drive/folders/1-aADgu93SDutxhjZQpT5ARD_a6msxSNS?usp=drive_link)

Model Architecture

<p align="center"> <img src="assets/hqd_em_architecture.jpg" alt="HQD-EM Architecture" width="850"> </p> Key ideas
HQD (Hierarchical Question Decomposition): Decomposes complex questions into sub-queries (e.g., subject → attribute → relation) to mitigate language bias.

EM (Ensemble + Margin): Ensembles models with different backbones/training seeds and expands the decision boundary using an Adaptive Angular Margin.
