import sys
import json
sys.path.insert(0, './resources/')
import constant
from model_utils import metric_dicts
import numpy as np
from eval_metric import mrr
import random

with open(constant.FILE_ROOT + 'crowd/dev.json') as f:
    line_elems = [json.loads(sent.strip()) for sent in f.readlines()]
    left_seq = [line_elem['left_context_token'] for line_elem in line_elems]
    mention_seq = [line_elem["mention_span"].split() for line_elem in line_elems]
    right_seq = [line_elem['right_context_token'] for line_elem in line_elems]
    seqs = [i+j+k for i,j,k in list(zip(left_seq, mention_seq, right_seq))]
    y_str_list = [line_elem['y_str'] for line_elem in line_elems]



# index = random.sample(range(len(seqs)), 1)[0]
# print(' '.join(seqs[index]))
# print(' '.join(mention_seq[index]))
# print(y_str_list[index])

gold_and_pred = json.load(open('best_predictions.json'))
# gold_and_pred = json.load(open('nogcn_predictions.json'))
# probs = np.load('nointer_probs.npy')
# y = np.load('nointer_y.npy')

general_types = set(['person', 'group', 'organization', 'location', 'entity', 'time', 'object', 'place', 'event'])

# # error analysis 
# ps = []
# rs = []
# for true, pred in gold_and_pred:
#     if pred:
#         trueset = set(true) - general_types
#         predset = set(pred) - general_types
#         if len(trueset) == 0:
#             continue
#         if len(predset) == 0:
#             ps.append(0)
#             rs.append(0)
#             continue

#         p = len(predset.intersection(trueset)) / float(len(predset))
#         r = len(predset.intersection(trueset)) / float(len(trueset))
#         ps.append(p)
#         rs.append(r)
#     else:
#         print('empty')
    
# print(np.mean(ps))
# print(rs)

# find the samples with pronouns
pronouns_results = []
else_results = []
pronoun_probs = []
else_probs = []
pronoun_ys = []
else_ys = []
pronouns_set = set(['he', 'I', 'they', 'him', 'it', 
    'himself', 'we','she', 'her', 'me', 'you', 'me', 'us', 'them', 'you', 'themselves','itself'])
for index, mention in enumerate(mention_seq):
    true, pred = gold_and_pred[index]
    trueset = set(true) - general_types
    predset = set(pred) - general_types
    if ' '.join(mention).strip().lower() in pronouns_set:
        pronouns_results.append([list(trueset), list(predset)])
        # pronoun_probs.append(probs[index,:])
        # pronoun_ys.append(y[index,:])
    else:
        else_results.append([list(trueset), list(predset)])
        # else_probs.append(probs[index,:])
        # else_ys.append(y[index,:])

# print(pronoun_ys)

_, output = metric_dicts(gold_and_pred)
print('overall:', output)

_, output = metric_dicts(pronouns_results)
print('pronouns:', output)

_, output = metric_dicts(else_results)
print('else:', output)
