from collections import namedtuple, defaultdict

def load_vocab_dict(vocab_file_name, vocab_max_size=None, start_vocab_count=None):
  with open(vocab_file_name) as f:
    text = [x.strip() for x in f.readlines()]
    if vocab_max_size:
      text = text[:vocab_max_size]
    if start_vocab_count:
      file_content = dict(zip(text, range(0 + start_vocab_count, len(text) + start_vocab_count)))
    else:
      file_content = dict(zip(text, range(0, len(text))))
  return file_content

FILE_ROOT = 'data/release/'
GLOVE_VEC = 'data/glove.840B.300d.txt'
FASTTEXT_WIKI_VEC = ''
FASTTEXT_CRAWL_VEC = ''
EXP_ROOT = 'exp/'
EXP_ROOT_ONTO = 'exp_onto/'

ANSWER_NUM_DICT = {"open": 10331, "onto":89, "wiki": 4600, "kb":130, "gen":9}

KB_VOCAB = load_vocab_dict(FILE_ROOT + "/ontology/types.txt", 130)
WIKI_VOCAB = load_vocab_dict(FILE_ROOT + "/ontology/types.txt", 4600)
ANSWER_VOCAB = load_vocab_dict(FILE_ROOT + "/ontology/types.txt")
ONTO_ANS_VOCAB = load_vocab_dict(FILE_ROOT + '/ontology/onto_ontology.txt')
ANS2ID_DICT = {"open": ANSWER_VOCAB, "wiki": WIKI_VOCAB, "kb": KB_VOCAB, "onto":ONTO_ANS_VOCAB}

open_id2ans = {v: k for k, v in ANSWER_VOCAB.items()}
wiki_id2ans = {v: k for k, v in WIKI_VOCAB.items()}
kb_id2ans = {v:k for k,v in KB_VOCAB.items()}
g_id2ans = {v: k for k, v in ONTO_ANS_VOCAB.items()}

ID2ANS_DICT = {"open": open_id2ans, "wiki": wiki_id2ans, "kb": kb_id2ans, "onto":g_id2ans}
label_string = namedtuple("label_types", ["head", "wiki", "kb"])
LABEL = label_string("HEAD", "WIKI", "KB")

CHAR_DICT = defaultdict(int)
char_vocab = [u"<unk>"]
with open(FILE_ROOT + "/ontology/char_vocab.english.txt") as f:
  char_vocab.extend(c.strip() for c in f.readlines())
  CHAR_DICT.update({c: i for i, c in enumerate(char_vocab)})

import json

pronouns_set = set(['he', 'I', 'they', 'him', 'it', 
    'himself', 'we','she', 'her', 'me', 'you', 'me', 'us', 'them', 'you', 'themselves','itself'])

pronoun_index_dev = []
else_index_dev = []
with open(FILE_ROOT + 'crowd/dev.json') as f:
    line_elems = [json.loads(sent.strip()) for sent in f.readlines()]
    mention_seq = [line_elem["mention_span"].split() for line_elem in line_elems]

for index, mention in enumerate(mention_seq):
  if ' '.join(mention).strip().lower() in pronouns_set:
    pronoun_index_dev.append(index) 
  else:
    else_index_dev.append(index)

pronoun_index_test = []
with open(FILE_ROOT + 'crowd/test.json') as f:
    line_elems = [json.loads(sent.strip()) for sent in f.readlines()]
    mention_seq = [line_elem["mention_span"].split() for line_elem in line_elems]

for index, mention in enumerate(mention_seq):
  if ' '.join(mention).strip().lower() in pronouns_set:
    pronoun_index_test.append(index) 

if __name__ == '__main__':
  print(len(pronoun_index_dev))