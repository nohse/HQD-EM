import argparse
import json
import os
from classifier import SimpleClassifier
import torch
from torch.utils.data import DataLoader
from fc import FCNet, MLP
from bc import BCNet
# from dataset import Dictionary, VQAFeatureDataset
from counting import Counter
import base_model
from base_model import GenB, Discriminator,GenBi, BanModelB
from language_model import WordEmbedding, QuestionEmbedding
from attention import Attention, NewAttention, BiAttention
from train import train
import utils
import click
from utils.dataset import Dictionary, VQAFeatureDataset
import utils.utils as utils
from utils.losses import Plain

import pdb


def parse_args():
    parser = argparse.ArgumentParser("Train the BottomUpTopDown model with a de-biasing method")

    # Arguments we added
    parser.add_argument('--cache_features', default=False, help="Cache image features in RAM. Makes things much faster"
                        "especially if the filesystem is slow, but requires at least 48gb of RAM")
    parser.add_argument('--dataset', default='cpv1', choices=["v2", "cpv2", "cpv1"], help="Run on VQA-2.0 instead of VQA-CP 2.0")
    parser.add_argument('--eval_each_epoch', default=True,help="Evaluate every epoch, instead of at the end")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='logs/exp0')
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--load_checkpoint_path', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset=args.dataset
    args.output=os.path.join('logs',args.output)
    # if not os.path.isdir(args.output):
    #     utils.create_dir(args.output)
    # else:
    #     if click.confirm('Exp directory already exists in {}. Erase?'
    #                              .format(args.output, default=False)):
    #         os.system('rm -r ' + args.output)
    #         utils.create_dir(args.output)

    #     else:
    #         if args.load_checkpoint_path is None:
    #             os._exit(1)


    dictionary = Dictionary.load_from_file('data/dictionary.pkl')

    print("Building train dataset...")
    train_dset = VQAFeatureDataset('train', dictionary)

    print("Building test dataset...")
    eval_dset = VQAFeatureDataset('val', dictionary)

    # Build the model using the original constructor
    constructor = 'build_%s' % args.model
    model, margin_model = getattr(base_model, constructor)(train_dset, args.num_hid)
    model = model.cuda()
    margin_model = margin_model.cuda()
    w_emb = WordEmbedding(train_dset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, args.num_hid, 1, False, 0.0)
    v_att = BiAttention(train_dset.v_dim, args.num_hid, args.num_hid, 4)
    b_net = []
    q_prj = []
    c_prj = []
    objects = 10  # minimum number of boxes
    num_hid=args.num_hid
    for i in range(4):
        b_net.append(BCNet(train_dset.v_dim, args.num_hid, args.num_hid, None, k=1))
        q_prj.append(FCNet([num_hid, num_hid], '', .2))
        c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, train_dset.num_ans_candidates, .5)
    counter = Counter(objects)
    genb= BanModelB(train_dset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, ' ', 4, num_hid=1024).cuda()


    # genbi = GenBi(num_hid=1024, dataset=train_dset).cuda()
    discriminator = Discriminator(num_hid=1024, dataset=train_dset).cuda()
    # discriminatori = Discriminator(num_hid=1024, dataset=train_dset).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    genb.w_emb.init_embedding('data/glove6b_init_300d.npy')
    # genbi.w_emb.init_embedding('data/glove6b_init_300d.npy')
    with open('util/qid2type_%s.json'%args.dataset,'r') as f:
        qid2type=json.load(f)

    if args.load_checkpoint_path is not None:
        ckpt = torch.load(os.path.join('logs', args.load_checkpoint_path, 'model.pth'))
        model_dict = model.state_dict()
        ckpt = {k: v for k, v in ckpt.items() if k in model_dict}
        model_dict.update(model_dict)
        model.load_state_dict(model_dict)

    batch_size = args.batch_size

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0)
    loss_fn = Plain()
    print("Starting training...")
    genbi=1
    discriminatori=1
    train(model, genb, discriminator, train_loader, eval_loader, args,qid2type, margin_model, loss_fn, genbi,discriminatori)

if __name__ == '__main__':
    main()
