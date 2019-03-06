import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable

from label_corr import build_concurr_matrix
import torch.nn.init as init

import eval_metric
sys.path.insert(0, './resources')
import constant

sigmoid_fn = nn.Sigmoid()


def get_eval_string(true_prediction):
  """
  Given a list of (gold, prediction)s, generate output string.
  """
  count, pred_count, avg_pred_count, p, r, f1 = eval_metric.micro(true_prediction)
  _, _, _, ma_p, ma_r, ma_f1 = eval_metric.macro(true_prediction)
  output_str = "Eval: {0} {1} {2:.3f} P:{3:.3f} R:{4:.3f} F1:{5:.3f} Ma_P:{6:.3f} Ma_R:{7:.3f} Ma_F1:{8:.3f}".format(
    count, pred_count, avg_pred_count, p, r, f1, ma_p, ma_r, ma_f1)
  accuracy = sum([set(y) == set(yp) for y, yp in true_prediction]) * 1.0 / len(true_prediction)
  output_str += '\t Dev accuracy: {0:.1f}%'.format(accuracy * 100)
  return output_str

def metric_dicts(true_prediction):
  count, pred_count, avg_pred_count, p, r, f1 = eval_metric.micro(true_prediction)
  _, _, _, ma_p, ma_r, ma_f1 = eval_metric.macro(true_prediction)
  output_str = "Eval: {0} {1} {2:.3f} P:{3:.3f} R:{4:.3f} F1:{5:.3f} Ma_P:{6:.3f} Ma_R:{7:.3f} Ma_F1:{8:.3f}".format(
    count, pred_count, avg_pred_count, p, r, f1, ma_p, ma_r, ma_f1)
  accuracy = sum([set(y) == set(yp) for y, yp in true_prediction]) * 1.0 / len(true_prediction)
  output_str += '\t Dev accuracy: {0:.1f}%'.format(accuracy * 100)
  result = {"precision": p, "recall": r, 'f1': f1, "ma_precision": ma_p, "ma_recall": ma_r, "ma_f1": ma_f1, "accu": accuracy}
  return result, output_str


def fine_grained_eval(true_prediction):
  general_types = set(['person', 'group', 'organization', 'location', 'entity', 'time', 'object', 'place', 'event'])
  fine_results = []
  # for index in constant.pronoun_index_dev:
  for index in constant.else_index_dev:
    true, pred = true_prediction[index]
    # trueset = set(true) - general_types
    # predset = set(pred) - general_types
    trueset = set(true)
    predset = set(pred)
    fine_results.append([list(trueset), list(predset)])
  metrics, output = metric_dicts(fine_results)
  return metrics, output

def get_output_index(outputs, thresh=0.5):
  """
  Given outputs from the decoder, generate prediction index.
  :param outputs:
  :return:
  """
  pred_idx = []
  outputs = sigmoid_fn(outputs).data.cpu().clone()
  for single_dist in outputs:
    single_dist = single_dist.numpy()
    arg_max_ind = np.argmax(single_dist)
    pred_id = [arg_max_ind]
    pred_id.extend(
      [i for i in range(len(single_dist)) if single_dist[i] > thresh and i != arg_max_ind])
    pred_idx.append(pred_id)
  return pred_idx


def get_gold_pred_str(pred_idx, gold, goal):
  """
  Given predicted ids and gold ids, generate a list of (gold, pred) pairs of length batch_size.
  """
  id2word_dict = constant.ID2ANS_DICT[goal]
  gold_strs = []
  for gold_i in gold:
    gold_strs.append([id2word_dict[i] for i in range(len(gold_i)) if gold_i[i] == 1])
  pred_strs = []
  for pred_idx1 in pred_idx:
    pred_strs.append([(id2word_dict[ind]) for ind in pred_idx1])
  return list(zip(gold_strs, pred_strs))

def sort_batch_by_length(tensor: torch.autograd.Variable, sequence_lengths: torch.autograd.Variable):
  """
  @ from allennlp
  Sort a batch first tensor by some specified lengths.

  Parameters
  ----------
  tensor : Variable(torch.FloatTensor), required.
      A batch first Pytorch tensor.
  sequence_lengths : Variable(torch.LongTensor), required.
      A tensor representing the lengths of some dimension of the tensor which
      we want to sort by.

  Returns
  -------
  sorted_tensor : Variable(torch.FloatTensor)
      The original tensor sorted along the batch dimension with respect to sequence_lengths.
  sorted_sequence_lengths : Variable(torch.LongTensor)
      The original sequence_lengths sorted by decreasing size.
  restoration_indices : Variable(torch.LongTensor)
      Indices into the sorted_tensor such that
      ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
  """

  if not isinstance(tensor, Variable) or not isinstance(sequence_lengths, Variable):
    raise ValueError("Both the tensor and sequence lengths must be torch.autograd.Variables.")

  sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
  sorted_tensor = tensor.index_select(0, permutation_index)
  # This is ugly, but required - we are creating a new variable at runtime, so we
  # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
  # refilling one of the inputs to the function.
  index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
  # This is the equivalent of zipping with index, sorting by the original
  # sequence lengths and returning the now sorted indices.
  index_range = Variable(index_range.long())
  _, reverse_mapping = permutation_index.sort(0, descending=False)
  restoration_indices = index_range.index_select(0, reverse_mapping)
  return sorted_tensor, sorted_sequence_lengths, restoration_indices


class MultiSimpleDecoder(nn.Module):
  """
    Simple decoder in multi-task setting.
  """
  def __init__(self, output_dim):
    super(MultiSimpleDecoder, self).__init__()
    self.linear = nn.Linear(output_dim, constant.ANSWER_NUM_DICT['open'],
                            bias=False).cuda()  # (out_features x in_features)

  def forward(self, inputs, output_type):
    if output_type == "open":
      return self.linear(inputs)
    elif output_type == 'wiki':
      return F.linear(inputs, self.linear.weight[:constant.ANSWER_NUM_DICT['wiki'], :], self.linear.bias)
    elif output_type == 'kb':
      return F.linear(inputs, self.linear.weight[:constant.ANSWER_NUM_DICT['kb'], :], self.linear.bias)
    else:
      raise ValueError('Decoder error: output type not one of the valid')

class SimpleDecoder(nn.Module):
  def __init__(self, output_dim, answer_num):
    super(SimpleDecoder, self).__init__()
    self.answer_num = answer_num
    self.linear = nn.Linear(output_dim, answer_num, bias=False)

  def forward(self, inputs, output_type):
    output_embed = self.linear(inputs)
    return output_embed

class GCNSimpleDecoder(nn.Module):
  """for ontonotes"""
  def __init__(self, output_dim, answer_num, goal="onto"):
    super(GCNSimpleDecoder, self).__init__()
    self.answer_num = answer_num
    self.linear = nn.Linear(output_dim, answer_num, bias=False)

    # gcn on label vectors
    self.transform = nn.Linear(output_dim, output_dim, bias=False)
    self.label_matrix, _, _, _, _ = build_concurr_matrix(goal=goal)
    self.label_matrix = torch.from_numpy(self.label_matrix).to(torch.device('cuda')).float()

    init.xavier_normal_(self.transform.weight)
    init.xavier_normal_(self.linear.weight)

  def forward(self, inputs, output_type):
    connection_matrix = self.label_matrix 
    transform = self.transform(connection_matrix.mm(self.linear.weight)  / connection_matrix.sum(1, keepdim=True))
    label_vectors = transform + self.linear.weight # residual
    logits = F.linear(inputs, label_vectors, self.linear.bias)
    return logits


class GCNMultiDecoder(nn.Module):
  """for ultra-fine"""
  def __init__(self, output_dim):
    super(GCNMultiDecoder, self).__init__()
    self.output_dim = output_dim
    self.linear = nn.Linear(output_dim, constant.ANSWER_NUM_DICT['open'], bias=False)
    self.transform = nn.Linear(output_dim, output_dim, bias=False)
    
    label_matrix, affinity, _, _, _ = build_concurr_matrix()
    self.label_matrix = torch.from_numpy(label_matrix).to(torch.device('cuda')).float()

    self.affinity = (torch.from_numpy(affinity).to(torch.device('cuda')).float() + 1) / 2
    self.weight = nn.Parameter(torch.rand(1))

  def forward(self, inputs, output_type):

    connection_matrix = self.label_matrix + self.weight * self.affinity
    label_vectors = self.transform(connection_matrix.mm(self.linear.weight) / connection_matrix.sum(1, keepdim=True))

    if output_type == "open":
      return self.linear(inputs)
    elif output_type == 'wiki':
      return F.linear(inputs, label_vectors[:constant.ANSWER_NUM_DICT['wiki'], :], self.linear.bias)
    elif output_type == 'kb':
      return F.linear(inputs, label_vectors[:constant.ANSWER_NUM_DICT['kb'], :], self.linear.bias)
    else:
      raise ValueError('Decoder error: output type not one of the valid')

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1d = nn.Conv1d(100, 50, 5)  # input, output, filter_number
    self.char_W = nn.Embedding(115, 100)

  def forward(self, span_chars):
    char_embed = self.char_W(span_chars).transpose(1, 2)  # [batch_size, char_embedding, max_char_seq]
    conv_output = [self.conv1d(char_embed)]  # list of [batch_size, filter_dim, max_char_seq, filter_number]
    conv_output = [F.relu(c) for c in conv_output]  # batch_size, filter_dim, max_char_seq, filter_num
    cnn_rep = [F.max_pool1d(i, i.size(2)) for i in conv_output]  # batch_size, filter_dim, 1, filter_num
    cnn_output = torch.squeeze(torch.cat(cnn_rep, 1), 2)  # batch_size, filter_num * filter_dim, 1
    return cnn_output


class DotAttn(nn.Module):
  """docstring for DotAttn"""
  def __init__(self, hidden_dim):
    super(DotAttn, self).__init__()
    self.hidden_dim = hidden_dim
    self.attn = nn.Linear(hidden_dim, hidden_dim, bias=False)

  def normalize(self, raw_scores, lens):
      backup = raw_scores.data.clone()
      max_len = raw_scores.size(1)

      for i, length in enumerate(lens):
          if length == max_len:
              continue
          raw_scores.data[i, int(length):] = -1e8

      normalized_scores = F.softmax(raw_scores, dim=-1)
      raw_scores.data.copy_(backup)
      return normalized_scores

  def forward(self, key, memory, lengths):
    '''
    key (bsz, hidden)
    memory (bsz, seq_len, hidden)
    '''
    scores = torch.bmm(memory, self.attn(key).unsqueeze(2)).squeeze(2)
    attn_scores = self.normalize(scores, lengths)
    retrieved = torch.sum(attn_scores.unsqueeze(2) * memory, dim=1) # (bsz, hidden)
    return retrieved
    
class SelfAttentiveSum(nn.Module):
  """
  Attention mechanism to get a weighted sum of RNN output sequence to a single RNN output dimension.
  """

  def __init__(self, output_dim, hidden_dim):
    super(SelfAttentiveSum, self).__init__()
    self.key_maker = nn.Linear(output_dim, hidden_dim, bias=False)
    self.key_rel = nn.ReLU()
    self.hidden_dim = hidden_dim
    self.key_output = nn.Linear(hidden_dim, 1, bias=False)
    self.key_softmax = nn.Softmax(dim=-1)

  def normalize(self, raw_scores, lens):
      backup = raw_scores.data.clone()
      max_len = raw_scores.size(1)

      for i, length in enumerate(lens):
          if length == max_len:
              continue
          raw_scores.data[i, int(length):] = -1e8

      normalized_scores = F.softmax(raw_scores, dim=-1)
      raw_scores.data.copy_(backup)
      return normalized_scores

  def forward(self, input_embed, input_lens=None):
    input_embed_squeezed = input_embed.view(-1, input_embed.size()[2])
    k_d = self.key_maker(input_embed_squeezed)
    k_d = self.key_rel(k_d) # (b*seq_len, hidden_dim)
    if self.hidden_dim == 1:
      k = k_d.view(input_embed.size()[0], -1)
    else:
      k = self.key_output(k_d).view(input_embed.size()[0], -1)  # (batch_size, seq_length)

    # normalize for seq_len
    if input_lens is not None:
      weighted_keys = self.normalize(k, input_lens).view(input_embed.size()[0], -1, 1)
    else:
      weighted_keys = self.key_softmax(k).view(input_embed.size()[0], -1, 1)
    weighted_values = torch.sum(weighted_keys * input_embed, 1)  # batch_size, seq_length, embed_dim
    return weighted_values, weighted_keys
