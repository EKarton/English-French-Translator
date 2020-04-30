import torch
import numpy as np


def get_embedding_weight_matrix(
    vocab_size, id2word, word_embedding_size, embedding_file
):
    # Get the word2vec from the embeddings filepath
    word2vec = {}
    iamdone = False
    for line in embedding_file:
        line = line.split()
        word = line[0]
        vect = np.array(line[1:]).astype(np.float)

        # Check if the embedding is the same size
        if vect.shape[0] != word_embedding_size:
            raise ValueError(
                "Word embedding for {} has dimension {}, not {}.".format(
                    word, vect.shape[0], word_embedding_size
                )
            )
        
        word2vec[word] = vect

    print("File contains {} word embeddings".format(len(word2vec)))

    # Create our word embedding matrix
    weights_matrix = np.zeros((vocab_size, word_embedding_size))
    words_found = 0

    for id_ in id2word:
        word = id2word[id_]
        if word in word2vec:
            weights_matrix[id_] = word2vec[word]
            words_found += 1

        else:
            weights_matrix[id_] = np.random.normal(
                scale=0.6, size=(word_embedding_size,)
            )

    print("Num words found:", words_found)
    return torch.from_numpy(weights_matrix)
