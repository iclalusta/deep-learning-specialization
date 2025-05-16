import numpy as np
from w2v_utils import *
words, word_to_vector_map = read_glove_vecs('glove.6B.50d.txt')

def cosine_similarity(u, v):
    distance = 0.0
    dot = np.dot(u, v)
    norm_u = np.sqrt(np.sum(u*u))
    norm_v = np.sqrt(np.sum(v*v))
    cosine_similarity = dot / (norm_u * norm_v)
    return cosine_similarity

"""father = word_to_vector_map["father"]
mother = word_to_vector_map["mother"]
ball = word_to_vector_map["ball"]
#crocodile = word_to_vector_map["crocodile"]
france = word_to_vector_map["france"]
italy = word_to_vector_map["italy"]
paris = word_to_vector_map["paris"]
rome = word_to_vector_map["rome"]
community = word_to_vector_map["community"]
shield = word_to_vector_map["shield"]
gay = word_to_vector_map["gay"]
lesbian = word_to_vector_map["lesbian"]
woman = word_to_vector_map["woman"]
man = word_to_vector_map["man"]

print("cosine similarity (father, mother) = ", cosine_similarity(father, mother))
#print("cosine similarity (ball, crocodile) = ", cosine_similarity(ball, crocodile))
print("cosine similarity (france-paris, rome-italy) = ", cosine_similarity(france-paris, italy-rome))
print("cosine similarity (community, shield) = ", cosine_similarity(community, shield))
print("cosine similarity (gay, lesbian) = ", cosine_similarity(gay, lesbian))
print("cosine similarity (mother, lesbian) = ", cosine_similarity(mother, lesbian))
print("cosine similarity (father, lesbian) = ", cosine_similarity(father, lesbian))
print("cosine similarity (man, woman) = ", cosine_similarity(man, woman))
print("cosine similarity (woman, lesbian) = ", cosine_similarity(woman, lesbian))
print("cosine similarity (man, lesbian) = ", cosine_similarity(man, lesbian))
print("cosine similarity (woman, gay) = ", cosine_similarity(woman, gay))
print("cosine similarity (man, gay) = ", cosine_similarity(man, gay))"""

def complete_analogy(word_a, word_b, word_c, word_to_vector_map):
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    e_a, e_b, e_c = word_to_vector_map[word_a], word_to_vector_map[word_b], word_to_vector_map[word_c]
    words = word_to_vector_map.keys()
    max_cosine_sim = -100
    best_word = None
    input_words_set = set([word_a, word_b, word_c])
    for w in words:
        if w in input_words_set:
            continue
        cos_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)
        if cos_sim > max_cosine_sim:
            max_cosine_sim = cos_sim
            best_word= w
    return best_word

"""triads_to_try = [('italy', 'italian', 'turkey'), ('book', 'writer', 'journal'), ('woman', 'mother', 'man'), ('gay', 'man', 'lesbian')]
for triad in triads_to_try:
    print('{} -> {} :: {} -> {}'.format(*triad, complete_analogy(*triad, word_to_vector_map)))"""

g = word_to_vector_map['woman'] - word_to_vector_map['man']
print(g)

"""print ('List of names and their similarities with constructed vector:')
name_list = ['john', 'marie', 'ronaldo', 'daniel', 'ray', 'austin', 'taylor', 'charlie', 'julia']
for w in name_list:
    print(w, cosine_similarity(word_to_vector_map[w], g))
print('Other words and their similarities:')
word_list = ['guns', 'science', 'arts', 'literature', 'war','doctor', 'tree', 'reception',
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w, cosine_similarity(word_to_vector_map[w], g))"""

def neutralize(word, g, word_to_vector_map):
    e = word_to_vector_map[word]
    e_biascomponent = np.dot(e, g) / np.sum(g*g) * g
    e_debiased = e - e_biascomponent
    return e_debiased

def equalize(pair, bias_axis, word_to_vector_map):
    w1, w2 = pair
    e_w1, e_w2 = word_to_vector_map[w1], word_to_vector_map[w2]
    mu = (e_w1 + e_w2) / 2
    mub = np.dot(mu, bias_axis) / np.sum(bias_axis * bias_axis) * bias_axis
    mut = mu - mub
    e_w1b = np.dot(e_w1, bias_axis) / np.sum(bias_axis * bias_axis) * bias_axis
    e_w2b = np.dot(e_w2, bias_axis) / np.sum(bias_axis * bias_axis) * bias_axis
    e_w1correctedb = np.sqrt(np.abs(1 - np.sum(mut * mut))) * (e_w1b - mub) / np.linalg.norm((e_w1 - mut) - mub)
    e_w2correctedb = np.sqrt(np.abs(1 - np.sum(mut * mut))) * (e_w2b - mub) / np.linalg.norm((e_w2 - mut) - mub)
    e1 = e_w1correctedb + mut
    e2 = e_w2correctedb + mut
    return e1, e2

"""print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vector_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vector_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vector_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
"""

