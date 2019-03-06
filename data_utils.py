"""A library for loading Type Dataset."""
import glob
import json
import logging
import random
import sys
from collections import defaultdict
import gluonnlp

import numpy as np

sys.path.insert(0, './resources/')
import constant
import torch




def to_torch(feed_dict):
  torch_feed_dict = {}
  if 'annot_id' in feed_dict:
    annot_ids = feed_dict.pop('annot_id')
  for k, v in feed_dict.items():
    if 'embed' in k:
      torch_feed_dict[k] = torch.autograd.Variable(torch.from_numpy(v), requires_grad=False).cuda().float()
    elif 'elmo' in k:
      torch_feed_dict[k] = v
    elif 'token_bio' == k:
      torch_feed_dict[k] = torch.autograd.Variable(torch.from_numpy(v), requires_grad=False).cuda().float()
    elif 'y' == k or k == 'mention_start_ind' or k == 'mention_end_ind' or 'length' in k:
      torch_feed_dict[k] = torch.autograd.Variable(torch.from_numpy(v), requires_grad=False).cuda()
    elif k == 'span_chars':
      torch_feed_dict[k] = torch.autograd.Variable(torch.from_numpy(v), requires_grad=False).cuda()
    elif k == 'token_seq_mask':
      torch_feed_dict[k] = torch.from_numpy(v).byte().cuda()

    elif k == 'context' or k == 'mention':
      torch_feed_dict[k] = v

    else:
      torch_feed_dict[k] = torch.from_numpy(v).cuda()
  return torch_feed_dict, annot_ids


def load_embedding_dict(embedding_path, embedding_size):
  print("Loading word embeddings from {}...".format(embedding_path))
  default_embedding = np.zeros(embedding_size)
  embedding_dict = defaultdict(lambda: default_embedding)
  with open(embedding_path) as f:
    for i, line in enumerate(f.readlines()):
      splits = line.split()
      if len(splits) != embedding_size + 1:
        continue
      assert len(splits) == embedding_size + 1
      word = splits[0]
      embedding = np.array([float(s) for s in splits[1:]])
      embedding_dict[word] = embedding
  print("Done loading word embeddings!")
  return embedding_dict

def get_vocab(source='glove'):
  """
  Get vocab file [word -> embedding]
  """
  char_vocab = constant.CHAR_DICT
  if source == 'glove':
    word_vocab = load_embedding_dict(constant.GLOVE_VEC, 300)
  elif source == 'fasttext_wiki':
    word_vocab = load_embedding_dict(constant.FASTTEXT_WIKI_VEC, 300)
  elif source == 'fasttext_crawl':
    word_vocab = load_embedding_dict(constant.FASTTEXT_CRAWL_VEC, 300)

  return char_vocab, word_vocab


def pad_slice(seq, seq_length, cut_left=False, pad_token="<none>"):
  if len(seq) >= seq_length:
    if not cut_left:
      return seq[:seq_length]
    else:
      output_seq = [x for x in seq if x != pad_token]
      if len(output_seq) >= seq_length:
        return output_seq[-seq_length:]
      else:
        return [pad_token] * (seq_length - len(output_seq)) + output_seq
  else:
    return seq + ([pad_token] * (seq_length - len(seq)))


def get_word_vec(word, vec_dict):
  if word in vec_dict:
    return vec_dict[word]
  return vec_dict['unk']


def build_vocab(file_list = ['crowd/dev.json', 'crowd/train_m.json', 'crowd/test.json', 'ontonotes/augmented_train.json', 'ontonotes/g_dev.json', 'ontonotes/g_test.json', 'distant_supervision/headword_train.json', 'distant_supervision/headword_dev.json', 'distant_supervision/el_dev.json', 'distant_supervision/el_train.json']):
  data_path = "data/release/"
  words = []
  for file in file_list:
    file_name = data_path + file
    with open(file_name) as f:
      line_elems = [json.loads(sent.strip()) for sent in f.readlines()]
      mention_seq = [line_elem["mention_span"].split() for line_elem in line_elems]
      left_seq = [line_elem['left_context_token'] for line_elem in line_elems]
      right_seq = [line_elem['right_context_token'] for line_elem in line_elems]
      for _ in mention_seq + right_seq + left_seq:
        words += [tok.lower() for tok in _]
  counter = gluonnlp.data.count_tokens(words)
  vocab = gluonnlp.Vocab(counter)
  with open('data/release/idx_to_token', 'w') as g:
    g.write('\n'.join(vocab.idx_to_token))
  with open('data/release/token_to_idx.json', 'w') as g:
    json.dump(vocab.token_to_idx, g)

def load_vocab():
  with open('data/release/idx_to_token') as f:
    idx_to_token = [word.strip() for word in f.readlines()]
  with open('data/release/token_to_idx.json') as g:
    token_to_idx = json.load(g)
  return idx_to_token, token_to_idx



def get_example(generator, glove_dict, batch_size, answer_num,
                eval_data=False, lstm_type="two", simple_mention=True):
  embed_dim = 300
  cur_stream = [None] * batch_size
  no_more_data = False

  while True:
    bsz = batch_size
    seq_length = 25
    for i in range(batch_size):
      try:
        cur_stream[i] = list(next(generator))
      except StopIteration:
        no_more_data = True
        bsz = i
        break
    if lstm_type == "two":
      left_embed = np.zeros([bsz, seq_length, embed_dim], np.float32)
      right_embed = np.zeros([bsz, seq_length, embed_dim], np.float32)
      left_seq_length = np.zeros([bsz], np.int32)
      right_seq_length = np.zeros([bsz], np.int32)
    else:
      max_seq_length = min(50, max([len(elem[1]) + len(elem[2]) + len(elem[3]) for elem in cur_stream if elem]))
      token_embed = np.zeros([bsz, max_seq_length, embed_dim], np.float32)
      token_seq_mask = np.ones([bsz, max_seq_length])
      token_seq_length = np.zeros([bsz], np.float32)
      token_bio = np.zeros([bsz, max_seq_length, 4], np.float32)
      token_bio_mask = np.zeros([bsz, max_seq_length], np.float32)
      mention_len = np.zeros([bsz], np.float32)
      mention_start_ind = np.zeros([bsz, 1], np.int64)
      mention_end_ind = np.zeros([bsz, 1], np.int64)

    max_mention_length = min(20, max([len(elem[3]) for elem in cur_stream if elem]))
    max_span_chars = min(25, max(max([len(elem[5]) for elem in cur_stream if elem]), 5))
    annot_ids = np.zeros([bsz], np.object)
    span_chars = np.zeros([bsz, max_span_chars], np.int64)
    mention_embed = np.zeros([bsz, max_mention_length, embed_dim], np.float32)
    targets = np.zeros([bsz, answer_num], np.float32)

    context = []
    mention = []

    for i in range(bsz):
      left_seq = cur_stream[i][1]
      if len(left_seq) > seq_length:
        left_seq = left_seq[-seq_length:]
      mention_seq = cur_stream[i][3]
      annot_ids[i] = cur_stream[i][0]
      right_seq = cur_stream[i][2]

      mention.append(' '.join(mention_seq))
      context.append(' '.join(left_seq + mention_seq + right_seq))

      # SEPARATE LSTM SETTING for left / right
      if lstm_type == "two":
        left_seq_length[i] = max(1, min(len(cur_stream[i][1]), seq_length))
        right_seq_length[i] = max(1, min(len(cur_stream[i][2]), seq_length))
        start_j = max(0, seq_length - len(left_seq))
        for j, left_word in enumerate(left_seq):
          if j < seq_length:
            left_embed[i, start_j + j, :300] = get_word_vec(left_word, glove_dict)
        for j, right_word in enumerate(cur_stream[i][2]):
          if j < seq_length:
            right_embed[i, j, :300] = get_word_vec(right_word, glove_dict)
      # SINGLE LSTM
      else:
        token_seq = left_seq + mention_seq + right_seq
        mention_start_ind[i] = min(seq_length, len(left_seq))
        mention_end_ind[i] = min(49, len(left_seq) + len(mention_seq) - 1)
        for j, word in enumerate(token_seq):
          if j < max_seq_length:
            token_embed[i, j, :300] = get_word_vec(word, glove_dict)
        for j, _ in enumerate(left_seq):
          token_bio[i, min(j, 49), 0] = 1.0  # token bio: 0(left) start(1) inside(2)  3(after)
          token_bio_mask[i, min(j, 49)] = 0.0
        for j, _ in enumerate(right_seq):
          token_bio[i, min(j + len(mention_seq) + len(left_seq), 49), 3] = 1.0
          token_bio_mask[i, min(j + len(mention_seq) + len(left_seq), 49)] = 0.0
        for j, _ in enumerate(mention_seq):
          if j == 0 and len(mention_seq) == 1:
            token_bio[i, min(j + len(left_seq), 49), 1] = 1.0
          else:
            token_bio[i, min(j + len(left_seq), 49), 2] = 1.0
          token_bio_mask[i,  min(j + len(left_seq), 49)] = 1.0

        token_seq_length[i] = min(50, len(token_seq))


        if token_seq_length[i] < 50:
          token_seq_mask[i, int(token_seq_length[i]):] = 0

        mention_len[i] = min(len(mention_seq), max_mention_length)

      for j, mention_word in enumerate(mention_seq):
        if j < max_mention_length:
          if simple_mention:
            mention_embed[i, j, :300] = [k / len(cur_stream[i][3]) for k in
                                         get_word_vec(mention_word, glove_dict)]
          else:
            mention_embed[i, j, :300] = get_word_vec(mention_word, glove_dict)
      span_chars[i, :] = pad_slice(cur_stream[i][5], max_span_chars, pad_token=0)
      for answer_ind in cur_stream[i][4]:
        targets[i, answer_ind] = 1.0

    feed_dict = {"annot_id": annot_ids,
                 "mention_embed": mention_embed,
                 "span_chars": span_chars,
                 "y": targets}

    if lstm_type == "two":
      feed_dict["right_embed"] = np.flip(right_embed, 1).copy()
      feed_dict["left_embed"] = left_embed
      feed_dict["right_seq_length"] = right_seq_length
      feed_dict["left_seq_length"] = left_seq_length
    else:
      feed_dict["token_bio"] = token_bio
      feed_dict["token_embed"] = token_embed
      feed_dict["token_seq_length"] = token_seq_length
      feed_dict["token_seq_mask"] = token_seq_mask
      feed_dict["mention_start_ind"] = mention_start_ind
      feed_dict["mention_end_ind"] = mention_end_ind
      feed_dict["token_bio_mask"] = token_bio_mask
      feed_dict["mention_len"] = mention_len


      # for analysis
      feed_dict['context'] = context
      feed_dict['mention'] = mention

    if no_more_data:
      if eval_data and bsz > 0:
        yield feed_dict
      break
    yield feed_dict


class TypeDataset(object):
  """Utility class type datasets"""

  def __init__(self, filepattern, vocab, goal, lstm_type):
    """Initialize Type Vocabulary
    Args:
      filepattern: Dataset file pattern.
      vocab: Vocabulary.
    """
    self._all_shards = glob.glob(filepattern)
    self.goal = goal
    self.lstm_type = lstm_type
    self.answer_num = constant.ANSWER_NUM_DICT[goal]
    random.shuffle(self._all_shards)
    self.char_vocab, self.glove_dict = vocab
    self.word2id = constant.ANS2ID_DICT[goal]
    print("Answer num %d" % (self.answer_num))
    print('Found %d shards at %s' % (len(self._all_shards), filepattern))
    logging.info('Found %d shards at %s' % (len(self._all_shards), filepattern))

  def _load_shard(self, shard_name, eval_data):
    """Read one file and convert to ids.
    Args:
      shard_name: file path.
    Returns:
      list of (id, global_word_id) tuples.
    """
    with open(shard_name) as f:
      line_elems = [json.loads(sent.strip()) for sent in f.readlines()]
      if not eval_data:
        line_elems = [line_elem for line_elem in line_elems if len(line_elem['mention_span'].split()) < 11]
      annot_ids = [line_elem["annot_id"] for line_elem in line_elems]
      mention_span = [[self.char_vocab[x] for x in list(line_elem["mention_span"])] for line_elem in line_elems]
      mention_seq = [line_elem["mention_span"].split() for line_elem in line_elems]
      left_seq = [line_elem['left_context_token'] for line_elem in line_elems]
      right_seq = [line_elem['right_context_token'] for line_elem in line_elems]
      y_str_list = [line_elem['y_str'] for line_elem in line_elems]
      y_ids = []
      for iid, y_strs in enumerate(y_str_list):
        y_ids.append([self.word2id[x] for x in y_strs if x in self.word2id])
    return zip(annot_ids, left_seq, right_seq, mention_seq, y_ids, mention_span)

  def _get_sentence(self, epoch, forever, eval_data, shuffle=False):
    for i in range(0, epoch if not forever else 100000000000000):
      for shard in self._all_shards:
        ids = list(self._load_shard(shard, eval_data))
        if shuffle:
          # print('Shuffle training data')
          np.random.shuffle(ids)
        for current_ids in ids:
          yield current_ids

  def get_batch(self, batch_size=128, epoch=5, forever=False, eval_data=False, simple_mention=True, shuffle=False):
    return get_example(self._get_sentence(epoch, forever=forever, eval_data=eval_data, shuffle=shuffle), self.glove_dict,
                       batch_size=batch_size, answer_num=self.answer_num, eval_data=eval_data,
                       simple_mention=simple_mention, lstm_type=self.lstm_type)

if __name__ == '__main__':
  build_vocab()
  # load_vocab()
