import pickle
import json
import os

# 입력 pkl 파일 경로
pkl_paths = {
    "ans2label": "/home/sejong/seonghyeon/newrml/data/cp-cache/trainval_ans2label.pkl",
    "label2ans": "/home/sejong/seonghyeon/newrml/data/cp-cache/trainval_label2ans.pkl",
}

# 출력 JSON 파일 경로 (같은 디렉토리에 저장)
json_paths = {
    "ans2label": os.path.splitext(pkl_paths["ans2label"])[0] + ".json",
    "label2ans": os.path.splitext(pkl_paths["label2ans"])[0] + ".json",
}

for key in ("ans2label", "label2ans"):
    # 1) Pickle 로드
    with open(pkl_paths[key], "rb") as f:
        data = pickle.load(f)
    # 2) JSON으로 직렬화할 때, key가 정수인 경우 문자열로 변환
    if key == "label2ans":
        data = {str(k): v for k, v in data.items()}
    # 3) JSON 파일로 저장
    with open(json_paths[key], "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {key} → {json_paths[key]}")
