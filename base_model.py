import torch
import torch.nn as nn
from attention import Attention, NewAttention, BiAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet, MLP
from bc import BCNet
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from counting import Counter
import torch.nn.init as init

import utils.config as config
import math

import pdb


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


# class BaseModel(nn.Module):
#     def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier):
#         super(BaseModel, self).__init__()
#         self.w_emb = w_emb
#         self.q_emb = q_emb
#         self.v_att = v_att
#         self.q_net = q_net
#         self.v_net = v_net
#         self.classifier = classifier

#     def forward(self, v, q):
#         """Forward
#         v: [batch, num_objs, obj_dim]
#         b: [batch, num_objs, b_dim]
#         q: [batch_size, seq_length]
#         return: logits
#         """
#         w_emb = self.w_emb(q)        
#         q_emb, _ = self.q_emb(w_emb)  # [batch, q_dim]

#         att = self.v_att(v, q_emb)

#         att = nn.functional.softmax(att, 1)
#         v_emb = (att * v).sum(1)  # [batch, v_dim]      

#         q_repr = self.q_net(q_emb)
#         v_repr = self.v_net(v_emb)
        
#         joint_repr = v_repr * q_repr
#         logits = self.classifier(joint_repr)

#         return logits, joint_repr
class BanModel(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, glimpse):
        super(BanModel, self).__init__()
        self.dataset = dataset
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        boxes = b[:,:,:4].transpose(1,2)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h
            
            atten, _ = logits[:,g,:,:].max(2)
            embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)
        joint_repr = q_emb.sum(1)
        logits = self.classifier(q_emb.sum(1))

        return logits, att, joint_repr

class BanModelB(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, op, glimpse, num_hid):
        super(BanModelB, self).__init__()
        self.dataset = dataset
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

        # v 생성용 generator
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.generate = nn.Sequential(
            *block(num_hid // 8, num_hid // 4),
            *block(num_hid // 4, num_hid // 2),
            *block(num_hid // 2, num_hid),
            nn.Linear(num_hid, num_hid * 2),
            nn.ReLU(inplace=True)
        )

    def select_random_tokens(self, tokens, num_tokens):
        batch_size, seq_len, emb_dim = tokens.shape
        mask = torch.zeros(seq_len, dtype=torch.bool, device=tokens.device)
        mask[0] = True
        mask[1] = True
        if num_tokens > 2:
            remaining = seq_len - 2
            select_count = num_tokens - 2
            selected_indices = torch.randperm(remaining, device=tokens.device)[:select_count] + 2
            mask[selected_indices] = True
        return tokens[:, mask, :]

    def forward(self, v, b, q, labels, flag=0, gen=True):
        w_emb = self.w_emb(q)
        batch_size, num_objs, obj_dim  = v.shape
        z = torch.cuda.FloatTensor(batch_size * num_objs, 128).normal_(0, 1)
        v = self.generate(z).view(batch_size, num_objs, obj_dim)
        if flag == 1:
            # noise 기반 v 생성


            # 질문 분해: 전체 + 절반 + 1/3
            q_emb1 = self.q_emb.forward_all(w_emb)

            w_emb2 = self.select_random_tokens(w_emb, (w_emb.shape[1] // 2) - 1)
            q_emb2 = self.q_emb.forward_all(w_emb2)

            w_emb3 = self.select_random_tokens(w_emb, (w_emb.shape[1] // 3) - 1)
            q_emb3 = self.q_emb.forward_all(w_emb3)

            # Step 1. 목표 시퀀스 길이 구하기
            target_len = q_emb1.size(1)  # = 14

            # Step 2. 각 시퀀스 패딩
            def pad_to(tensor, target_len):
                pad_len = target_len - tensor.size(1)
                if pad_len > 0:
                    # pad는 (dim_last, dim_first) 순서로 지정
                    tensor = F.pad(tensor, (0, 0, 0, pad_len), "constant", 0)
                return tensor

            q_emb2 = pad_to(q_emb2, target_len)  # (512, 14, 1024)
            q_emb3 = pad_to(q_emb3, target_len)  # (512, 14, 1024)

            # Step 3. 세 시퀀스를 합치기 (sum 또는 concat)
            q_emb = q_emb1 + q_emb2 + q_emb3  # 또는 torch.cat([q_emb1, q_emb2, q_emb3], dim=1)

        else:
            # 기존 방식 유지
            q_emb = self.q_emb.forward_all(w_emb)

        # BAN 기본 처리
        boxes = b[:, :, :4].transpose(1, 2)
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb)

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:, g, :, :])
            atten, _ = logits[:, g, :, :].max(2)
            embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = self.classifier(q_emb.sum(1))
        return logits, att



class GenB(nn.Module):
    def __init__(self, num_hid, dataset):
        super(GenB, self).__init__()
        self.num_hid = num_hid
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
        self.q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        self.v_att1 = NewAttention(dataset.v_dim, self.q_emb.num_hid, num_hid)
        self.v_att2 = NewAttention(dataset.v_dim, self.q_emb.num_hid, num_hid)
        self.v_att3 = NewAttention(dataset.v_dim, self.q_emb.num_hid, num_hid)
        self.q_net = FCNet([self.q_emb.num_hid, num_hid])
        self.q_net2 = FCNet([self.q_emb.num_hid, num_hid])
        self.q_net3 = FCNet([self.q_emb.num_hid, num_hid])
        self.v_net = FCNet([dataset.v_dim, num_hid])
        self.classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
        self.flag=1

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.generate = nn.Sequential(
            *block(num_hid // 8, num_hid // 4),
            *block(num_hid // 4, num_hid // 2),
            *block(num_hid // 2, num_hid),
            nn.Linear(num_hid, num_hid * 2),
            nn.ReLU(inplace=True)
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def select_random_tokens(self, tokens, num_tokens):
        batch_size, seq_len, emb_dim = tokens.shape
        # Create a boolean mask for token selection
        mask = torch.zeros(seq_len, dtype=torch.bool, device=tokens.device)
        # Always keep the first two tokens
        mask[0] = True
        mask[1] = True
        # Select the remaining tokens randomly if needed
        if num_tokens > 2:
            remaining = seq_len - 2
            select_count = num_tokens - 2
            # Randomly pick indices from the tokens after the first two
            selected_indices = torch.randperm(remaining, device=tokens.device)[:select_count] + 2
            mask[selected_indices] = True
        # Preserve original order and filter tokens
        return tokens[:, mask, :]
    def select_tokens_with_stride(self, tokens, stride):
        """
        주어진 stride에 따라 특정 간격으로 토큰을 선택하는 함수
        """
        return tokens[:, ::stride, :]  # stride 간격으로 토큰 선택


    def forward(self, v, q, flag, gen=True):
        w_emb = self.w_emb(q)
        total_tokens = w_emb.shape[1]
        if flag == 1:
            # 1. 전체 문장
            q_embw, _ = self.q_emb(w_emb)
            att1 = self.v_att1(v, q_embw)
            q_emb1 = self.q_net(q_embw)

            # 2. 절반 크기의 랜덤 토큰
            w_emb2 = self.select_random_tokens(w_emb, (total_tokens // 2)-1)
            q_emb2, _ = self.q_emb(w_emb2)
            att2 = self.v_att1(v, q_emb2)
            q_emb2 = self.q_net2(q_emb2)

            # 3. 3분의 1 크기의 랜덤 토큰
            w_emb3 = self.select_random_tokens(w_emb, (total_tokens // 3)-1)
            q_emb3, _ = self.q_emb(w_emb3)
            att3 = self.v_att1(v, q_emb3)
            q_emb3 = self.q_net3(q_emb3)
            # Element-wise product
            q_repr = q_emb1+q_emb2 + q_emb3
            b, c, f = v.shape


            # Generate from noise
            if gen:
                v_z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (b, c, 128))))
                v = self.generate(v_z.view(-1, 128)).view(b, c, f)

            att=att1+att2+att3
            att = F.softmax(att, 1)
            v_emb = (att * v).sum(1)
            v_repr = self.v_net(v_emb)
            joint_repr = v_repr * q_repr

            logits = self.classifier(joint_repr)
            return logits

        else:
            w_emb = self.w_emb(q)
            q_emb, _ = self.q_emb(w_emb)
            b, c, f = v.shape

            # generate from noise
            if gen:
                v_z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (b, c, 128))))
                v = self.generate(v_z.view(-1, 128)).view(b, c, f)

            att = self.v_att1(v, q_emb)
            att = nn.functional.softmax(att, 1)
            v_emb = (att * v).sum(1)

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)

            joint_repr = v_repr * q_repr

            logits = self.classifier(joint_repr)

            return logits

class GenBi(nn.Module):
    def __init__(self, num_hid, dataset):
        super(GenBi, self).__init__()
        self.num_hid = num_hid
        self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
        self.ntoken = dataset.dictionary.ntoken  # 18455
        self.q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        self.v_att = NewAttention(dataset.v_dim, self.q_emb.num_hid, num_hid)
        self.q_net = FCNet([self.q_emb.num_hid, num_hid])
        self.v_net = FCNet([dataset.v_dim, num_hid])
        self.classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.generate = nn.Sequential(
            *block(num_hid // 8, num_hid // 4),
            *block(num_hid // 4, num_hid // 2),
            *block(num_hid // 2, num_hid),
            nn.Linear(num_hid, num_hid * 2),
            nn.ReLU(inplace=True)
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, v, q, flag, gen=True):
        b, c, f = v.shape

        # gen가 True인 경우, q의 shape와 동일한 무작위 q를 생성합니다.
        # 여기서 q의 shape는 (512, 14)이며, 각 원소는 0 ~ 18454 범위의 정수입니다.
        if gen:
            q = torch.randint(0, self.ntoken, q.shape, device=q.device)

        w_emb = self.w_emb(q)
        q_emb, _ = self.q_emb(w_emb)

        att = self.v_att(v, q_emb)
        att = nn.functional.softmax(att, dim=1)
        v_emb = (att * v).sum(1)

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)

        joint_repr = v_repr * q_repr

        logits = self.classifier(joint_repr)

        return logits



# class GenB(nn.Module):
#     def __init__(self, num_hid, dataset):
#         super(GenB, self).__init__()
#         self.num_hid = num_hid
#         self.w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
#         self.q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
#         self.v_att = NewAttention(dataset.v_dim, self.q_emb.num_hid, num_hid)
#         self.q_net = FCNet([self.q_emb.num_hid, num_hid])
#         self.v_net = FCNet([dataset.v_dim, num_hid])
#         self.classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)

#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#         self.generate = nn.Sequential(
#             *block(num_hid//8, num_hid//4),
#             *block(num_hid//4, num_hid//2),
#             *block(num_hid//2, num_hid),
#             nn.Linear(num_hid, num_hid*2),
#             nn.ReLU(inplace=True)
#             )
#         self.weight_init()

#     def weight_init(self):
#         for block in self._modules:
#             try:
#                 for m in self._modules[block]:
#                     kaiming_init(m)
#             except:
#                 kaiming_init(block)

#     def forward(self, v, q, gen=True):
#         w_emb = self.w_emb(q)
#         q_emb, _ = self.q_emb(w_emb)

#         b, c, f = v.shape

#         # generate from noise
#         if gen==True:
#             v_z = Variable(torch.cuda.FloatTensor(np.random.normal(0,1, (b,c, 128))))
#             v = self.generate(v_z.view(-1, 128)).view(b,c,f)

#         att = self.v_att(v, q_emb)

#         att = nn.functional.softmax(att, 1)
#         v_emb = (att * v).sum(1)

#         q_repr = self.q_net(q_emb)
#         v_repr = self.v_net(v_emb)

#         joint_repr = v_repr * q_repr

#         logits = self.classifier(joint_repr)

#         return logits


class Discriminator(nn.Module):
    def __init__(self, num_hid, dataset):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dataset.num_ans_candidates, 1024),
            nn.ReLU(True),
            nn.Linear(num_hid, num_hid//2),
            nn.ReLU(True),
            nn.Linear(num_hid//2, num_hid//4),
            nn.ReLU(True),
            nn.Linear(num_hid//4, 1),
            nn.Sigmoid(),
            )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)



class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=config.scale, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.std = 0.1
        self.temp = config.temp

    def forward(self, input, learned_mg, m, epoch, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if self.training is False:
            return None, cosine
        
        #Set beta (Subsecion 3.3 in main paper
        beta_factor = epoch // 15
        beta = 1.0 - (beta_factor * 0.1)

        #Calculate the learnable instance-level margins, Subsection 3.3 in main paper
        learned_mg = torch.where(m > 1e-12, learned_mg.double(), -1000.0).float()

        margin = F.softmax(learned_mg / self.temp, dim=1)
        margin=margin
        # Perform randomization as mentioned in Section 3 of main paper
        if config.randomization:
            m = torch.normal(mean=m, std=self.std)
            
        #Combine the margins, as in Subsection 3.3 of main paper.
        if config.learnable_margins:
            m[label != 0] = beta * m[label != 0] + (1 - beta) * margin[label != 0]
        m = 1 - m

        #Compute the AdaArc angular margins and the corresponding logits
        self.cos_m = torch.cos(m)
        self.sin_m = torch.sin(m)
        self.th = torch.cos(math.pi - m)
        self.mm = torch.sin(math.pi - m) * m
        # --------------------------- cos(theta) & phi(theta) ---------------------------

        # cosine = input
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = phi * self.s
        return output, cosine

class BiasAdjustedArcMarginProduct(nn.Module):
    r"""Bias-Adjusted ArcMarginProduct: 기존 ArcMarginProduct의 흐름을 유지하면서,
    genb의 logit (bias_score)을 이용해 margin에 선형 보정을 적용합니다.
    
    Args:
        in_features (int): 입력 feature 차원.
        out_features (int): 출력 클래스 수.
        s (float): feature 스케일링 계수 (default: config.scale).
        lambda_bias (float): bias 신호에 곱해질 가중치 (기본 0.5).
        easy_margin (bool): easy margin 사용 여부.
    """
    def __init__(self, in_features, out_features, s=config.scale, lambda_bias=0.5, easy_margin=False):
        super(BiasAdjustedArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.lambda_bias = lambda_bias
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.std = 0.1
        self.temp = config.temp

    def forward(self, input, learned_mg, m, epoch, label, bias_score):
        """
        Args:
            input (Tensor): 입력 feature, shape [batch, in_features].
            learned_mg (Tensor): 학습 가능한 margin 값.
            m (Tensor): 기본 margin (원래 코드에서 사용하던 값).
            epoch (int): 현재 epoch (dynamic scaling 적용용).
            label (Tensor): ground truth 레이블, shape [batch].
            bias_score (Tensor): 각 샘플별 bias 정도 (genb의 logit, shape: [batch]).
        Returns:
            output (Tensor): bias 보정이 적용된 로짓, shape [batch, out_features].
            cosine (Tensor): cosine 유사도 값.
        """
        # 1. cosine similarity 계산 (정규화 후 내적)
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if not self.training:
            return None, cosine

        # 2. beta 계산 (epoch에 따라 감소)
        beta_factor = epoch // 15
        beta = 1.0 - (beta_factor * 0.1)

        # 3. 학습 가능한 margin 조정 (같은 dtype 유지)
        learned_mg = torch.where(m > 1e-12, learned_mg, torch.full_like(m, -1000.0))
        margin = F.softmax(learned_mg / self.temp, dim=1)
        if config.randomization:
            m = torch.normal(mean=m, std=self.std)
        if config.learnable_margins:
            m[label != 0] = beta * m[label != 0] + (1 - beta) * margin[label != 0]
        # 4. bias_score 반영: genb logit 값을 sigmoid로 확률로 변환
        bias_prob = torch.sigmoid(bias_score)  # [batch]
        m = m + self.lambda_bias * bias_prob.unsqueeze(1)
        # 5. 최종 margin
        m = 1 - m

        # 6. angular margin 계산 (지역 변수 사용)
        cos_m = torch.cos(m)
        sin_m = torch.sin(m)
        th = torch.cos(math.pi - m)
        mm = torch.sin(math.pi - m) * m

        sine = torch.sqrt((1.0 - cosine * cosine).clamp(0, 1))
        phi = cosine * cos_m - sine * sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > th, phi, cosine - mm)

        output = phi * self.s
        return output, cosine

# -------------------------------------------------------------------
# 기존 build_baseline0_newatt 함수에서 margin_model 부분을 아래와 같이 수정합니다.
# def build_baseline0_newatt(dataset, num_hid):
#     w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
#     q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
#     v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
#     q_net = FCNet([q_emb.num_hid, num_hid])
#     v_net = FCNet([dataset.v_dim, num_hid])
#     classifier = SimpleClassifier(num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
#     basemodel = BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)
#     # 여기서 기존 ArcMarginProduct 대신 BiasAdjustedArcMarginProduct를 사용
#     margin_model = BiasAdjustedArcMarginProduct(num_hid, dataset.num_ans_candidates)
#     return basemodel, margin_model

def l2_norm(input, dim=-1):
    norm = torch.norm(input, dim=dim, keepdim=True)
    output = torch.div(input, norm)
    return output



def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)

def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = BiAttention(dataset.v_dim, num_hid, num_hid, 4)
    b_net = []
    q_prj = []
    c_prj = []
    objects = 10  # minimum number of boxes
    for i in range(4):
        b_net.append(BCNet(dataset.v_dim, num_hid, num_hid, None, k=1))
        q_prj.append(FCNet([num_hid, num_hid], '', .2))
        c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, .5)
    counter = Counter(objects)
    basemodel= BanModel(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, ' ', 4)
    margin_model = ArcMarginProduct(num_hid, dataset.num_ans_candidates)
    return basemodel, margin_model
