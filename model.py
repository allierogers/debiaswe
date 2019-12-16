import numpy as np
from debiaswe.data import load_professions
from debiaswe.debias import debias
from debiaswe.we import WordEmbedding

def compute_bias_direction(embedding, rep_words=['he', 'she']):
    """ Get the subspace that we will use to represent the bias. 
    """
    words_group1 = [rep_words[2 * i] for i in range(len(rep_words) // 2)]
    words_group2 = [rep_words[2 * i + 1] for i in range(len(rep_words) // 2)]
#     E = WordEmbedding('embeddings/w2v_gnews_small.txt')
    vs = [sum(embedding.v(w) for w in words) for words in (words_group2, words_group1)]
    vs = [v / np.linalg.norm(v) for v in vs]

    v_protected = vs[1] - vs[0]
    v_protected = v_protected / np.linalg.norm(v_protected)
    
    return v_protected

def compute_bias_scores(embedding, v_protected):
    # load professions
    professions = load_professions()
    profession_words = [p[0] for p in professions]
    
    # profession analysis gender
    sp = sorted([(embedding.v(w).dot(v_protected), w) for w in profession_words])

    return sp[0:20], sp[-20:]
    