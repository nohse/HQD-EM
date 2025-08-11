import os
import time

import torch
import torch.nn as nn
import utils.utils as utils
from torch.autograd import Variable
from tqdm import tqdm
import time
import torch.nn.functional as F
import pdb
Tensor = torch.cuda.FloatTensor


def compute_supcon_loss(feats, qtype):
    tau = 1.0
    if isinstance(qtype, tuple):
      i = 0
      dic = {}
      for item in qtype:
          if item not in dic:
              dic[item] = i
              i = i + 1
      tau = 1.0
      qtype = torch.tensor([dic[item] for item in qtype]).cuda()
    feats_filt = F.normalize(feats, dim=1)
    targets_r = qtype.reshape(-1, 1)
    targets_c = qtype.reshape(1, -1)
    mask = targets_r == targets_c
    mask = mask.int().cuda()
    feats_sim = torch.exp(torch.matmul(feats_filt, feats_filt.T) / tau)
    negatives = feats_sim*(1.0 - mask)
    negative_sum = torch.sum(negatives)
    positives = torch.log(feats_sim/negative_sum)*mask
    positive_sum = torch.sum(positives)
    positive_sum = positive_sum/torch.sum(mask)

    sup_con_loss = -1*torch.mean(positive_sum)
    return sup_con_loss


def compute_ga_supcon_loss(feats, gt, genb_logits, epoch, tau=1.0, lambda_supcon_base=1.0, warmup_epochs=5, gamma=2.0):
    """
    Adaptive SupCon Loss with amplified weighting and dynamic scaling.
    
    Args:
        feats (Tensor): Target model의 feature, shape [batch, feature_dim].
        gt (Tensor): Ground truth label, shape [batch], 각 원소는 정수 인덱스.
        genb_logits (Tensor): genb의 출력 logits, shape [batch, num_classes].
        epoch (int): 현재 epoch (dynamic한 scaling에 사용).
        tau (float): Temperature scaling factor (기본값: 1.0).
        lambda_supcon_base (float): SupCon loss의 기본 scaling factor.
        warmup_epochs (int): Warm-up에 사용할 epoch 수.
        gamma (float): Amplification exponent. (예: 2.0이면 확률을 제곱하여 강조)
    
    Returns:
        loss (Tensor): 최종 adaptive supcon loss.
    """
    # Dynamic scaling: 초기에는 λ 값을 낮게 시작하고 warmup_epochs 이후 1.0까지 증가
    current_lambda = lambda_supcon_base * min(1.0, epoch / warmup_epochs)
    # 1. genb의 logits에 sigmoid 적용하여 클래스별 확률로 변환 (각 샘플마다 독립적으로)
    prob_genb = torch.sigmoid(genb_logits)  # [batch, num_classes]
    # 각 샘플의 ground truth에 해당하는 확률 추출
    weights = prob_genb[torch.arange(genb_logits.size(0)), gt].detach()  # [batch]
    # Amplification: 높은 확률을 더욱 강조 (낮은 확률은 더 낮게)
    weights = weights ** gamma

    # 2. target feature 정규화 후 cosine 유사도 기반 logits 계산
    feats = F.normalize(feats, p=2, dim=1)
    logits = torch.matmul(feats, feats.T) / tau

    # 수치 안정성을 위해 각 행의 최대값을 빼줌
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # 3. ground truth 기준으로 positive pair 마스크 생성 (자기 자신 제외)
    mask = torch.eq(gt.unsqueeze(1), gt.unsqueeze(0)).float().cuda()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
    mask = mask * logits_mask

    # 4. 각 row에 대해 분모 계산 (자기 자신 제외)
    exp_logits = torch.exp(logits) * logits_mask
    sum_exp = exp_logits.sum(1, keepdim=True)  # [batch, 1]

    # 5. log 확률 계산
    log_prob = logits - torch.log(sum_exp + 1e-12)

    # 6. 각 샘플마다 positive pair들의 평균 log 확률 계산
    num_positives = mask.sum(1)
    mean_log_prob_pos = (mask * log_prob).sum(1) / (num_positives + 1e-12)

    # 7. 각 샘플의 loss (음의 평균 log_prob)
    loss_vec = -mean_log_prob_pos  # [batch]

    # 8. genb의 확률을 반영한 가중치 적용: 높은 weight일수록 loss 기여도 증가
    weighted_loss = weights * loss_vec

    # 9. 전체 loss는 가중치 합으로 나누고 동적 scaling factor 적용
    loss = current_lambda * weighted_loss.sum() / (weights.sum() + 1e-12)
    
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.argmax(logits, 1)
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def calc_genb_loss(logits, bias, labels):
    gen_grad = torch.clamp(2 * labels * torch.sigmoid(-2 * labels * bias.detach()), 0, 1)
    loss = F.binary_cross_entropy_with_logits(logits, gen_grad)
    loss *= labels.size(1)
    return loss

def train(model, genb, discriminator, train_loader, eval_loader, args, qid2type, margin_model, loss_fn, genbi, discriminatori):
    num_epochs = args.epochs
    run_eval = args.eval_each_epoch
    output = args.output

    # Optimizer 정의
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    optim_M = torch.optim.Adamax(filter(lambda p: p.requires_grad, margin_model.parameters()), lr=5e-4)
    optim_G = torch.optim.Adamax(filter(lambda p: p.requires_grad, genb.parameters()), lr=5e-4)
    optim_D = torch.optim.Adamax(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=5e-4)
    # (필요시 genbi, discriminatori용 옵티마이저도 추가)

    logger = utils.Logger(os.path.join(output, 'log.txt'))
    total_step = 0
    best_eval_score = 0
    best_epoch = 0

    model.train(True)
    margin_model.train(True)
    genb.train(True)
    discriminator.train(True)
    # genbi.train(True); discriminatori.train(True)  # 필요 시

    kld = nn.KLDivLoss(reduction='batchmean')
    bce = nn.BCELoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        train_score = 0.0
        t0 = time.time()
        flag = 1

        # -------------------------------------------------------------------
        # 배치 단위 루프 시작
        # -------------------------------------------------------------------
        for i, (v, q, a, mg, bias, q_id, f1, q_type, _, b) in tqdm(
                enumerate(train_loader), ncols=100,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                total=len(train_loader)):
            total_step += 1

            # 1) 입력 데이터를 GPU로 올리기
            v = v.cuda().requires_grad_()     # [batch, num_objs, obj_dim]
            q = q.cuda()                      # [batch, seq_len]
            a = a.cuda()                      # [batch, num_classes]
            b = b.cuda()                      # [batch, num_objs, b_dim]
            mg = mg.cuda()                    # [batch]
            bias = bias.cuda()                # [batch, num_classes] 또는 [batch, 1]
            f1 = f1.cuda()                    # [batch]
            valid = torch.ones((v.size(0), 1), device=v.device)
            fake = torch.zeros((v.size(0), 1), device=v.device)

            # -------------------------------------------------------------------
            # 2) Main 모델(BAN) 첫 번째 Forward
            #   → 이 forward에서 pred, joint_repr를 얻지만, 이 그래프는 곧 GenB→Model 단계에서 사용되고 해제됩니다.
            # -------------------------------------------------------------------
            pred, att, joint_repr = model(v, b, q, a)
            # pred: [B, num_ans_candidates], joint_repr: [B, num_hid]

            # -------------------------------------------------------------------
            # 3) Generator(genb) Forward (학습용)
            # -------------------------------------------------------------------
            pred_g, _ = genb(v, b, q, a, flag, gen=True)
            # pred_g: [B, num_ans_candidates]

            # -------------------------------------------------------------------
            # 4) Discriminator(D) 학습
            #   → pred, pred_g를 detach 상태로 넘겨야 “model/genb 그래프”와 단절됨
            # -------------------------------------------------------------------
            with torch.no_grad():
                pred_detach_for_D   = pred.detach()
                pred_g_detach_for_D = pred_g.detach()

            vae_preds = discriminator(pred_g_detach_for_D)  # [B,1]
            main_preds = discriminator(pred_detach_for_D)    # [B,1]

            d_loss = bce(vae_preds, fake) + bce(main_preds, valid)

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # -------------------------------------------------------------------
            # 5) Generator(genb) 학습
            # -------------------------------------------------------------------
            # 5.1) genb 자체 바이너리 CE loss
            g_bce = F.binary_cross_entropy_with_logits(pred_g, a, reduction='none').mean()
            g_bce = g_bce * a.size(1)

            # 5.2) D를 속이는 손실: pred → D(→valid), pred_g → D(→valid)
            #    여기서 pred는 detach(), pred_g는 원본(pred_g) 그대로
            vae_preds2 = discriminator(pred_g)            # [B,1] (pred_g은 그래프 연결)
            main_preds2 = discriminator(pred.detach())     # [B,1] (pred.detach()이므로 모델 그래프 X)
            g_d_loss = bce(vae_preds2, valid) + bce(main_preds2, valid)

            # 5.3) Distillation (KLDiv) loss: pred_g vs. pred.detach()
            g_distill = kld(
                F.log_softmax(pred_g, dim=1),
                F.softmax(pred.detach(), dim=1)
            )

            g_loss = g_bce + g_d_loss + g_distill * 5.0

            optim_G.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(genb.parameters(), 0.25)
            optim_G.step()

            # -------------------------------------------------------------------
            # 6) GenB → Main 모델 연결 학습
            # -------------------------------------------------------------------
            genb.train(False)
            with torch.no_grad():
                pred_g_frozen, _ = genb(v, b, q, a, flag, gen=False)
            genb.train(True)

            genb_loss = calc_genb_loss(pred, pred_g_frozen, a)
            # 여기서 pred_g_frozen은 이미 detach된 상태
            # pred(원본)는 여전히 “model 그래프”에 연결되어 있으므로, model→genb_loss 경로로 gradient 전파 가능

            optim.zero_grad()
            genb_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optim.step()

            # -------------------------------------------------------------------
            # 7) RML(Main) 학습: “새로운 Forward”를 수행해 fresh pred2, joint_repr2 얻기
            # -------------------------------------------------------------------
            # ↳ 여기서 다시 model(v,b,q,a) 호출 → 배치 내 모델 파라미터가 업데이트된 뒤의 fresh 그래프
            pred2, att2, joint_repr2 = model(v, b, q, a)
            # pred2: [B, num_ans], joint_repr2: [B, num_hid]

            hidden2, p12 = margin_model(joint_repr2, pred2, mg, epoch, a)
            # hidden2: [B, num_hid], p12: [B, num_ans_candidates]

            # 7.1) CE loss (RML 논문 방식)
            ce_loss = -F.log_softmax(pred2, dim=1) * a    # [B, num_ans]
            ce_loss = (ce_loss * f1.unsqueeze(1)).sum(dim=1).mean()

            # 7.2) margin‐based add-on loss
            dict_args = {'margin': mg, 'bias': bias, 'hidden': hidden2, 'epoch': epoch, 'per': f1}
            rml_margin_loss = loss_fn(hidden2, a, **dict_args)

            # 7.3) SupCon loss (batch 간 pairwise)
            gt_labels = torch.argmax(a, dim=1)  # [B]
            supcon_loss = compute_supcon_loss(joint_repr2, gt_labels)

            main_loss = ce_loss + rml_margin_loss + supcon_loss

            optim.zero_grad()
            optim_M.zero_grad()
            main_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            nn.utils.clip_grad_norm_(margin_model.parameters(), 0.25)
            optim.step()
            optim_M.step()

            # -------------------------------------------------------------------
            # 8) 배치별 손실/정확도 누적
            # -------------------------------------------------------------------
            total_loss += (genb_loss.item() + main_loss.item()) * v.size(0)

            with torch.no_grad():
                batch_score = compute_score_with_logits(p12, a).sum().item()
                train_score += batch_score

            # flag 토글
            flag = 0 if flag == 1 else 1

        # ===== 에폭(epoch) 종료 후 Logging & 평가 =====
        n_samples = len(train_loader.dataset)
        total_loss = total_loss / n_samples
        train_score = 100.0 * train_score / n_samples
        elapsed = time.time() - t0

        logger.write(f"Epoch {epoch+1}, time: {elapsed:.2f}s")
        logger.write(f"\ttrain_loss: {total_loss:.4f}, train_score: {train_score:.2f}")

        if run_eval:
            model.train(False)
            margin_model.train(False)

            results = evaluate(model, eval_loader, qid2type, margin_model, epoch)
            eval_score = results["score"]
            bound = results["upper_bound"]
            yn = results["score_yesno"]
            other = results["score_other"]
            num_ = results["score_number"]

            logger.write(f"\teval_score: {100*eval_score:.2f} (upper_bound: {100*bound:.2f})")
            logger.write(f"\tyn: {100*yn:.2f}, other: {100*other:.2f}, number: {100*num_:.2f}")

            if eval_score > best_eval_score:
                torch.save(model.state_dict(), os.path.join(output, "model.pth"))
                torch.save(genb.state_dict(), os.path.join(output, "genb.pth"))
                torch.save(margin_model.state_dict(), os.path.join(output, "margin_model.pth"))
                best_eval_score = eval_score
                best_epoch = epoch

            model.train(True)
            margin_model.train(True)

        # 매 에폭 마지막에 모델 “최종” 버전 저장
        torch.save(model.state_dict(), os.path.join(output, "model_final.pth"))

    print(f"Best eval score: {100*best_eval_score:.2f}, at epoch {best_epoch+1}")




def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss

def evaluate(model, dataloader, qid2type, margin_model, epoch=0):
    score = 0
    upper_bound = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0 
    temp = 0.3
    alpha = 0.5

    for v, q, a, mg, _, q_id, _, q_type, _, b in tqdm(dataloader, ncols=100, total=len(dataloader), desc="eval"):
        v = Variable(v, requires_grad=False).cuda()
        q = Variable(q, requires_grad=False).cuda()
        b=b.cuda()
        mg = mg.cuda()
        a = a.cuda()
        #genb evaluation
        # pred, _ = model(v, q)
        # rml evaluation
        ce_logits, att, hidden = model(v, b, q, a)
        hidden, pred = margin_model(hidden, ce_logits, mg, epoch, a, 1)

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
