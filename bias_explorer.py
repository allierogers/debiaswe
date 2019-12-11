from argparse import ArgumentParser

from debiaswe.we import WordEmbedding


def compute_bias_direction(rep_words):
    """ Get the subspace that we will use to represent the bias. 
    """
    words_group1 = [words[2 * i] for i in range(len(args.rep_words) // 2)]
    words_group2 = [words[2 * i + 1] for i in range(len(args.rep_words) // 2)]
    
    vs = [sum(E.v(w) for w in words) for words in (words_group2, words_group1)]
    vs = [v / np.linalg.norm(v) for v in vs]

    v_protected = vs[1] - vs[0]
    v_protected = v_protected / np.linalg.norm(v_protected)
    
    return v_protected


def main():
    parser = ArgumentParser()
    parser.add_argument('--we_path')
    parser.add_argument('--rep_words', default=None)
    args = parser.parse_args()
    
    # Load word embedding from provided filepath
    E = WordEmbedding(args.we_path)
    
    v_protected = compute_bias_direction(args.rep_words)

    # Analogies based on the protected direction
    a_protected = E.best_analogies_dist_thresh(v_protected)
