from __future__ import print_function
import os
import sys
import json
import numpy as np

# Add project root to path for importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary


def create_dictionary(dataroot):
    """
    Create a Dictionary from VQA question JSON files.
    """
    dictionary = Dictionary()
    files = [
        'OpenEnded_mscoco_train2014_questions.json',
        'OpenEnded_mscoco_val2014_questions.json',
        'OpenEnded_mscoco_test2015_questions.json'
        # Uncomment if using test-dev:
        # 'OpenEnded_mscoco_test-dev2015_questions.json'
    ]
    for fname in files:
        question_path = os.path.join(dataroot, fname)
        print(f"Loading questions from: {question_path}")
        with open(question_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for q in data.get('questions', []):
            dictionary.tokenize(q['question'], True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    """
    Initialize embedding matrix using pretrained GloVe vectors.
    """
    word2emb = {}
    # Read GloVe file entries
    with open(glove_file, 'r', encoding='utf-8') as f:
        entries = f.readlines()
    # Determine embedding dimension
    emb_dim = len(entries[0].split()) - 1
    print(f'embedding dim is {emb_dim}')
    # Initialize weights matrix
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    # Build word -> vector mapping
    for entry in entries:
        parts = entry.strip().split()
        word = parts[0]
        # Convert remaining strings to floats
        emb_vals = np.array([float(x) for x in parts[1:]], dtype=np.float32)
        word2emb[word] = emb_vals

    # Populate weights for words in dictionary
    for idx, word in enumerate(idx2word):
        vec = word2emb.get(word)
        if vec is not None:
            weights[idx] = vec
    return weights, word2emb


if __name__ == '__main__':
    # Define paths
    dataroot = '/home/mmai1/songhyeon/newrml/data'
    dict_path = os.path.join(dataroot, 'dictionary_v1.pkl')

    # Create and dump dictionary
    d = create_dictionary(dataroot)
    d.dump_to_file(dict_path)

    # Load dictionary and create embeddings
    d = Dictionary.load_from_file(dict_path)
    emb_dim = 300
    glove_file = os.path.join(dataroot, 'glove', f'glove.6B.{emb_dim}d.txt')
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)

    # Save initialized embeddings
    np.save(os.path.join(dataroot, f'glove6b_init_{emb_dim}d_v1.npy'), weights)
