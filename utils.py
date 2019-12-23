
import numpy as np

def load_word2vec(file, embed_dim=200):
    # load_word_vec
    vocab = dict()
    vocab['[PAD]'] = 0
    vocab['[UNK]'] = 1

    matrix = []
    matrix.append(np.zeros(embed_dim))
    matrix.append(np.random.random(embed_dim))
    
    vec_dim = 0
    with open(file, 'r') as fin:
        for l in fin.readlines():
            t = l.split(' ')
            if len(t) < 5:
                continue
            w = t[0]
            vec = t[1:]
            vec = [float(v) for v in vec]

            if vec_dim == 0:
                vec_dim = len(vec)
            
            vec = np.array(vec)
            cur_id = len(vocab)
            vocab[w] = cur_id
            matrix.append(vec)

    matrix = np.stack(matrix, 0)

    matrix = matrix.astype(np.float)
    
    return vocab, matrix


if __name__ == "__main__":
    file = 'data/tencent_embedding.txt'
    vocab, matrix = load_word2vec(file)
    print(len(vocab))
    print(matrix.shape)
    