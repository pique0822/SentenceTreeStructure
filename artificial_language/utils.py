import numpy as np

def character_to_vector(character,vocabulary):
    vector = np.zeros(len(vocabulary))
    idx = vocabulary.index(character)
    vector[idx] = 1

    return vector.reshape(1,-1), idx
