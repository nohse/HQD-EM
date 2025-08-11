import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
import numpy as np
from base_model import GenB
from utils.dataset import Dictionary, VQAFeatureDataset
import base_model
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm


def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def evaluate(model, dataloader, qid2type, margin_model, epoch=0, temp=1.0, alpha=1.0):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0

    for v, q, a, mg, _, q_id, _, q_type, _, b in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        b=Variable(b, requires_grad=False).cuda()
        mg = mg.cuda()
        a = a.cuda()

        # Forward pass
        ce_logits,att, hidden = model(v, b, q, a)
        # pred_g = genb(v, q, 1, gen=True)
        hidden, pred = margin_model(hidden, ce_logits, mg, epoch, a)

        ce_logits = F.softmax(F.normalize(ce_logits) / temp, 1)
        pred_l = F.softmax(F.normalize(pred), 1)
        pred = alpha * pred_l + (1-alpha) * ce_logits


        batch_score = compute_score_with_logits(pred, a.cuda()).cpu().numpy().sum(1)
        score += batch_score.sum()
        upper_bound += (a.max(1)[0]).sum()
        q_id = q_id.detach().cpu().int().numpy()
        for j in range(len(q_id)):
            qid = q_id[j]
            typ = qid2type[str(qid)]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')

        score = score / len(dataloader.dataset)
        upper_bound = upper_bound / len(dataloader.dataset)
        score_yesno /= total_yesno
        score_other /= total_other
        score_number /= total_number

        results = dict(
            score=score,
            upper_bound=upper_bound,
            score_yesno=score_yesno,
            score_other=score_other,
            score_number=score_number,
        )
        return results


def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")
    parser.add_argument('--cache_features', action='store_true',
                        help="Cache image features in RAM (requires ~48GB RAM)")
    parser.add_argument('--dataset', default='cpv1', choices=["v2", "cpv2", "cpv1"],
                        help="Run on VQA-2.0 instead of VQA-CP 2.0")
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--load_path', type=str, default='/home/mmai1/songhyeon/newrml/logs/logs/65.96(ban-cpv1)')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset

    # Load dictionary
    if dataset == 'cpv1':
        dictionary = Dictionary.load_from_file('data/dictionary_v1.pkl')
    else:
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    # Build datasets and models
    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary)
    constructor = f'build_{args.model}'
    model, margin_model = getattr(base_model, constructor)(eval_dset, args.num_hid)
    model, margin_model = model.cuda(), margin_model.cuda()
    # genb = GenB(num_hid=1024, dataset=eval_dset).cuda()
    # genb.w_emb.init_embedding('data/glove6b_init_300d.npy')

    with open(f'util/qid2type_{dataset}.json', 'r') as f:
        qid2type = json.load(f)

    # Load checkpoints
    model.load_state_dict(torch.load(os.path.join(args.load_path, 'model.pth')))
    margin_model.load_state_dict(torch.load(os.path.join(args.load_path, 'margin_model.pth')))
    # genb.load_state_dict(torch.load(os.path.join(args.load_path, 'genb.pth')))
    print('Loaded Model!')

    model.eval()
    margin_model.eval()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    eval_loader = DataLoader(eval_dset, args.batch_size, shuffle=False, num_workers=0)
    epoch = 3
    # Grid search over temp and alpha
    # temps = [i * 0.1 for i in range(1, 11)]
    # alphas = [i * 0.1 for i in range(0, 11)]
    # probs = [i * 0.1 for i in range(0, 11)]
    temps = [i * 0.1 for i in range(1, 11)]
    alphas = [i * 0.1 for i in range(1, 11)]
    probs = [1]
    best_score = -float('inf')
    best_temp, best_alpha = None, None

    for temp in temps:
        for alpha in alphas:
            # for prob in probs:
            results = evaluate(model, eval_loader, qid2type, margin_model, epoch, temp=temp, alpha=alpha)
            print(f"Score: {100 * results['score']:.2f}% with temp={temp} and alpha={alpha} and prob=0")
            if results['score'] > best_score:
                best_score = results['score']
                best_temp, best_alpha, best_prob = temp, alpha,0
                best_results = results

    print(f"Best Score: {100 * best_score:.2f}% with temp={best_temp} and alpha={best_alpha} and prob=0")
    print(f"  Yes/No score : {100 * best_results['score_yesno']:.2f}%")
    print(f"  Number score : {100 * best_results['score_number']:.2f}%")
    print(f"  Other score  : {100 * best_results['score_other']:.2f}%")

if __name__ == '__main__':
    main()
# # eval.py
# import argparse
# import json
# import os

# import torch
# from torch.utils.data import DataLoader
# import numpy as np
# from base_model import GenB
# from utils.dataset import Dictionary, VQAFeatureDataset
# import base_model
# import torch.nn.functional as F
# from torch.autograd import Variable
# from tqdm import tqdm

# # (추가) 이미지 시각화를 원하시면 uncomment
# # import matplotlib.pyplot as plt
# # from PIL import Image

# def compute_score_with_logits(logits, labels):
#     logits = torch.argmax(logits, 1)
#     one_hots = torch.zeros(*labels.size()).cuda()
#     one_hots.scatter_(1, logits.view(-1, 1), 1)
#     scores = (one_hots * labels)
#     return scores, logits  # logits 추가 반환

# def decode_question(q_tensor, dictionary):
#     words = []
#     vocab_size = len(dictionary.idx2word)
#     for idx in q_tensor.tolist():
#         if 0 <= idx < vocab_size:
#             words.append(dictionary.idx2word[idx])
#         else:
#             words.append("<unk>")
#     return ' '.join(words)


# def evaluate_qual(model, dataloader, dictionary, qid2type, margin_model,
#                   epoch=0, temp=1.0, alpha=0.2, topk=20):
#     corrects, wrongs = [], []
#     # 답안 인덱스 → 문자열 매핑
#     ans_list = dataloader.dataset.label2ans

#     # 전체/타입별 카운터 초기화
#     total_count = 0
#     correct_count = 0
#     type_stats = {'yes/no': [0, 0], 'number': [0, 0], 'other': [0, 0]}
#     print("*****")
#     print(temp)
#     print(alpha)
#     for v, q, a, mg, _, q_id, _, _ in tqdm(dataloader,
#                                           ncols=100, total=len(dataloader),
#                                           desc="qual_eval"):
#         v = v.cuda()
#         q = q.cuda()
#         mg = mg.cuda()
#         a = a.cuda()

#         # 1) base forward
#         ce_logits, hidden = model(v, q)
#         # 2) margin_model forward (genb_logits는 dummy)
#         _, pred_logits = margin_model(hidden, ce_logits, mg, epoch, a, 1)

#         # 3) combine
#         ce_probs   = F.softmax(F.normalize(ce_logits) / temp, dim=1)
#         pred_probs = F.softmax(F.normalize(pred_logits), dim=1)
#         combined   = alpha * pred_probs + (1 - alpha) * ce_probs

#         # 4) compute scores & preds
#         scores, preds = compute_score_with_logits(combined, a)
#         gts = torch.argmax(a, 1)  # ground-truth 인덱스

#         for i in range(v.size(0)):
#             total_count += 1
#             is_correct = (preds[i].item() == gts[i].item())
#             if is_correct:
#                 correct_count += 1

#             # q_type 대신 qid로부터 실제 타입을 꺼내옵니다.
#             qid = str(int(q_id[i].item()))
#             t = qid2type[qid]
#             type_stats[t][1] += 1
#             if is_correct:
#                 type_stats[t][0] += 1

#             entry = {
#                 'qid':      int(qid),
#                 'q_type':   t,
#                 'question': decode_question(q[i].cpu(), dictionary),
#                 'gt_ans':   int(gts[i].item()),
#                 'gt_str':   ans_list[gts[i].item()],
#                 'pred_ans': int(preds[i].item()),
#                 'pred_str': ans_list[preds[i].item()],
#                 'correct':  is_correct
#             }
#             if is_correct:
#                 corrects.append(entry)
#             else:
#                 wrongs.append(entry)

#     # 전체/타입별 accuracy 출력
#     print(f"\n>>> Overall Accuracy: {correct_count}/{total_count} = {correct_count/total_count:.4f}")
#     for typ, (c, n) in type_stats.items():
#         print(f"  {typ:6s} Accuracy: {c}/{n} = {c/n:.4f}")

#     # 상위 topk개만 보여주기
#     print(f"\n===== Top {topk} Correct Examples =====")
#     for e in corrects[:topk]:
#         print(f"QID={e['qid']} | type={e['q_type']} "
#               f"| GT={e['gt_ans']}[{e['gt_str']}] "
#               f"| PRED={e['pred_ans']}[{e['pred_str']}]")
#         print(f"  Q: {e['question']}\n")

#     print(f"\n===== Top {topk} Wrong Examples =====")
#     for e in wrongs[:topk]:
#         print(f"QID={e['qid']} | type={e['q_type']} "
#               f"| GT={e['gt_ans']}[{e['gt_str']}] "
#               f"| PRED={e['pred_ans']}[{e['pred_str']}]")
#         print(f"  Q: {e['question']}\n")

#     return {
#         'num_correct': len(corrects),
#         'num_wrong':   len(wrongs),
#     }



# def parse_args():
#     parser = argparse.ArgumentParser("Qualitative Ablation Study for VQA")
#     parser.add_argument('--dataset', default='cpv2', choices=["v2", "cpv2", "cpv1"])
#     parser.add_argument('--num_hid',  type=int,   default=1024)
#     parser.add_argument('--model',    type=str,   default='baseline0_newatt')
#     parser.add_argument('--batch_size', type=int, default=512)
#     parser.add_argument('--load_path',  type=str, default='/home/mmai1/songhyeon/newrml/logs/logs/62.53(1,0.2)')
#     parser.add_argument('--topk',       type=int, default=20,
#                         help="정/오답 샘플 몇 개씩 볼지")
#     return parser.parse_args()

# def main():
#     args = parse_args()
#     # Dictionary & Dataset
#     dictionary = Dictionary.load_from_file(
#         'data/dictionary_v1.pkl' if args.dataset=='cpv1' else 'data/dictionary.pkl'
#     )
#     eval_dset = VQAFeatureDataset('val', dictionary)
#     eval_loader = DataLoader(eval_dset, args.batch_size,
#                              shuffle=False, num_workers=4)

#     # 모델 로딩
#     constructor = f'build_{args.model}'
#     model, margin_model = getattr(base_model, constructor)(
#         eval_dset, args.num_hid
#     )
#     model.load_state_dict(torch.load(os.path.join(args.load_path,'model.pth')))
#     margin_model.load_state_dict(torch.load(os.path.join(args.load_path,'margin_model.pth')))
#     model, margin_model = model.cuda(), margin_model.cuda()
#     model.eval(); margin_model.eval()

#     # qid2type
#     with open(f'util/qid2type_{args.dataset}.json','r') as f:
#         qid2type = json.load(f)

#     # Qualitative evaluation
#     print("vocab size =", len(dictionary.idx2word))


#     stats = evaluate_qual(model, eval_loader, dictionary, qid2type,
#                           margin_model, epoch=0,
#                           temp=1.0, alpha=0.2, topk=args.topk)
#     print(f"\n>>> 총 correct: {stats['num_correct']}, wrong: {stats['num_wrong']}")

# if __name__ == '__main__':
#     main()

# import argparse
# import json
# import os

# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
# from tqdm import tqdm

# import base_model
# from base_model import GenB
# from utils.dataset import Dictionary, VQAFeatureDataset


# def compute_score_with_logits(logits, labels):
#     preds = torch.argmax(logits, dim=1)
#     one_hots = torch.zeros_like(labels).cuda()
#     one_hots.scatter_(1, preds.view(-1,1), 1)
#     scores = (one_hots * labels).sum(dim=1)
#     return scores, preds

# def evaluate_quant(model, dataloader, qid2type, margin_model, epoch=0, temp=1.0, alpha=1.0):
#     score = 0
#     upper_bound = 0
#     score_yesno = 0
#     score_number = 0
#     score_other = 0
#     total_yesno = 0
#     total_number = 0
#     total_other = 0
#     thresholds = list(range(4, 15))
#     length_counts  = {L: 0 for L in thresholds}  # “길이 ≥ L”인 샘플 수 누적
#     length_correct = {L: 0 for L in thresholds}  # “길이 ≥ L”인 샘플 중 정답 

#     for v, q, a, mg, _, q_id, _, q_type, _ in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
#         v = Variable(v, requires_grad=False).cuda()
#         q = Variable(q, requires_grad=False).cuda()
#         mg = mg.cuda()
#         a = a.cuda()

#         # Forward pass
#         ce_logits, hidden = model(v, q)
#         # pred_g = genb(v, q, 1, gen=True)
#         hidden, pred_logits = margin_model(hidden, ce_logits, mg, epoch, a, 1)

#         # Apply temperature and combine predictions
#         ce_probs = F.softmax(F.normalize(ce_logits) / temp, dim=1)
#         pred_probs = F.softmax(F.normalize(pred_logits), dim=1)
#         combined = alpha * pred_probs + (1 - alpha) * ce_probs

#         batch_score, preds = compute_score_with_logits(combined, a)
#         score += batch_score.sum()
#         upper_bound += (a.max(1)[0]).sum().item()

#         q_ids = q_id.detach().cpu().int().numpy()
#         for j, qid in enumerate(q_ids):
#             typ = qid2type[str(qid)]
#             if typ == 'yes/no':
#                 score_yesno += batch_score[j]
#                 total_yesno += 1
#             elif typ == 'other':
#                 score_other += batch_score[j]
#                 total_other += 1
#             elif typ == 'number':
#                 score_number += batch_score[j]
#                 total_number += 1
#         length_non_pad = (q != 18455).sum(dim=1)
#         for L in thresholds:
#             mask = (length_non_pad >= L)             # [batch], boolean tensor
#             cnt_L = int(mask.sum().item())           # 이 배치에서 길이 ≥ L인 샘플 수
#             if cnt_L > 0:
#                 # 그 위치에서 batch_scores[mask]를 더해 정답 수 누적
#                 correct_L = int(batch_score[mask].sum().item())
#                 length_counts[L]  += cnt_L
#                 length_correct[L] += correct_L

#         # 7) 카테고리별 정확도 누적 및 qualitative용 wrong 샘플 저장
#         gts   = torch.argmax(a, dim=1).cpu().numpy()  # [batch], 정답 클래스 인덱스
#     print("\n>>> Length‐Filtered Accuracy (padding 제외 토큰 수 기준)")
#     for L in thresholds:
#         cnt = length_counts[L]
#         if cnt > 0:
#             acc = length_correct[L] / cnt
#             print(f"  length >= {L:2d} : {length_correct[L]}/{cnt} = {acc:.4f}")
#         else:
#             print(f"  length >= {L:2d} : {length_correct[L]}/{cnt} = N/A (샘플 없음)")
#     # Compute averages
#     n = len(dataloader.dataset)
#     results = {
#         'score': score / n,
#         'upper_bound': upper_bound / n,
#         'score_yesno': score_yesno / total_yesno,
#         'score_other': score_other / total_other,
#         'score_number': score_number / total_number
#     }
#     return results


# def decode_question(q_tensor, dictionary):
#     words, V = [], len(dictionary.idx2word)
#     for idx in q_tensor.tolist():
#         words.append(dictionary.idx2word[idx] if 0<=idx< V else "<unk>")
#     return " ".join(words)


# def evaluate_qual(model, dataloader, dictionary, qid2type, margin_model,
#                   epoch=0, temp=1.0, alpha=1.0, topk=20):


#     # 1) 저장된 wrong QID 불러오기
#     with open("/home/mmai1/GenB/genBsh/wrong_qids.txt", 'r') as f:
#         saved_wrong = set(int(line.strip()) for line in f)

#     ans_list = dataloader.dataset.label2ans
#     filtered = []

#     for v, q, a, mg, _, q_id, _, _, i_id in tqdm(dataloader, desc="qual_eval"):
#         v, q, a = v.cuda(), q.cuda(), a.cuda()
#         mg = mg.cuda()

#         # 모델 출력
#         ce_logits, hidden = model(v, q)
#         _, pred_logits = margin_model(hidden, ce_logits, mg, epoch, a, 1)

#         ce_probs   = F.softmax(F.normalize(ce_logits) / temp, dim=1)
#         pred_probs = F.softmax(F.normalize(pred_logits),      dim=1)
#         combined   = alpha * pred_probs + (1 - alpha) * ce_probs

#         batch_scores, preds = compute_score_with_logits(combined, a)
#         gts = torch.argmax(a, 1)

#         for i in range(v.size(0)):
#             qid_val = int(q_id[i].item())
#             # 2) 저장된 wrong list 에 포함되어 있지 않으면 skip
#             if qid_val not in saved_wrong:
#                 continue

#             is_corr = (preds[i].item() == gts[i].item())
#             # 3) 저장된 wrong_qid.txt 는 “틀린” QID 리스트이므로, 
#             #    여기서는 실제로도 틀린 경우만 담습니다.
#             if not is_corr:
#                 continue

#             entry = {
#                 'qid':    qid_val,
#                 'img_id': int(i_id[i].item()),
#                 'q_type': qid2type[str(qid_val)],
#                 'question': decode_question(q[i].cpu(), dictionary),
#                 'gt_ans':   gts[i].item(),
#                 'gt_str':   ans_list[gts[i].item()],
#                 'pred_ans': preds[i].item(),
#                 'pred_str': ans_list[preds[i].item()],
#             }
#             filtered.append(entry)
#     output_path = "/home/mmai1/GenB/genBsh/filtered_corrected_qids.txt"
#     with open(output_path, 'w') as f:
#         for e in filtered:
#             f.write(f"{e['qid']}\n")
#     print(f"필터링된 QID {len(filtered)}개를 {output_path}에 저장했습니다.")

#     # 4) 필터링된 예시 출력
#     print(f"\n===== Filtered Wrong Examples (총 {len(filtered)}) =====")
#     for e in filtered[:topk]:
#         print(f"QID={e['qid']} | IMGID={e['img_id']} | type={e['q_type']} | "
#               f"GT={e['gt_ans']}[{e['gt_str']}] | "
#               f"PRED={e['pred_ans']}[{e['pred_str']}]")
#         print(f"  Q: {e['question']}\n")

#     return filtered



# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('--dataset',    default='cpv2', choices=['v2','cpv2','cpv1'])
#     p.add_argument('--model',      default='baseline0_newatt')
#     p.add_argument('--num_hid',    type=int,   default=1024)
#     p.add_argument('--batch_size', type=int,   default=512)
#     p.add_argument('--load_path',  default='/home/mmai1/songhyeon/newrml/logs/logs/65.96(ban-cpv1)')
#     p.add_argument('--topk',       type=int,   default=20)
#     return p.parse_args()



def main():
    args = parse_args()

    # -- data & model
    dict_path = 'data/dictionary_v1.pkl' if args.dataset=='cpv1' else 'data/dictionary.pkl'
    dictionary = Dictionary.load_from_file(dict_path)
    eval_dset  = VQAFeatureDataset('val', dictionary)
    eval_loader= DataLoader(eval_dset, args.batch_size, shuffle=False, num_workers=4)

    model, margin_model = getattr(base_model, f'build_{args.model}')(eval_dset, args.num_hid)
    model.load_state_dict(   torch.load(os.path.join(args.load_path,'model.pth')))
    margin_model.load_state_dict(torch.load(os.path.join(args.load_path,'margin_model.pth')))
    model,margin_model = model.cuda(), margin_model.cuda()
    model.eval(); margin_model.eval()

    with open(f'util/qid2type_{args.dataset}.json') as f:
        qid2type = json.load(f)

    # -- quantitative: grid search
    best, best_t, best_a = -1, None, None
    t,a=1,0.2
    for t in [i*0.1 for i in range(1,11)]:
        for a in [i*0.1 for i in range(11)]:
            results = evaluate_quant(model, eval_loader, qid2type, margin_model, 
                                            epoch=3, temp=t, alpha=a)
    print(f"t={t:.1f} a={a:.1f} -> Overall {100*results['score']:.2f}%, YN {100*results['score_yesno']:.2f}%, "
            f"Num {100*results['score_number']:.2f}%, Other {100*results['score_other']:.2f}%")
    if acc>best:
        best, best_t, best_a = acc, t, a

    print(f"\n>> Best quantitative: t={best_t:.1f}, a={best_a:.1f} overall={100*best:.2f}%")

    # # -- qualitative @ best hyperparams
    # print("\n######## Qualitative Examples ########")
    # filtered=evaluate_qual(model, eval_loader, dictionary, qid2type, margin_model,
    #               epoch=3, temp=1, alpha=0.2, topk=args.topk)


if __name__=='__main__':
    main()
