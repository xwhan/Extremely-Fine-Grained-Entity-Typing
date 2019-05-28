import sys

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import sort_batch_by_length, SelfAttentiveSum, SimpleDecoder, MultiSimpleDecoder, CNN, GCNMultiDecoder, GCNSimpleDecoder, DotAttn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from label_corr import build_concurr_matrix
import numpy as np
from attention import SimpleEncoder

sys.path.insert(0, './resources')
import constant

def cosine_similarity(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Fusion(nn.Module):
    """docstring for Fusion"""
    def __init__(self, d_hid):
        super(Fusion, self).__init__()
        self.r = nn.Linear(d_hid*3, d_hid)
        self.g = nn.Linear(d_hid*3, d_hid)

    def forward(self, x, y):
        r_ = gelu(self.r(torch.cat([x,y,x-y], dim=-1)))
        g_ = torch.sigmoid(self.g(torch.cat([x,y,x-y], dim=-1)))
        return g_ * r_ + (1 - g_) * x
    

class Model(nn.Module):
  def __init__(self, args, answer_num):
    super(Model, self).__init__()
    self.args = args
    self.output_dim = args.rnn_dim * 2
    self.mention_dropout = nn.Dropout(args.mention_dropout)
    self.input_dropout = nn.Dropout(args.input_dropout)
    self.dim_hidden = args.dim_hidden
    self.embed_dim = 300
    self.mention_dim = 300
    self.lstm_type = args.lstm_type
    self.enhanced_mention = args.enhanced_mention
    if args.enhanced_mention:
      self.head_attentive_sum = SelfAttentiveSum(self.mention_dim, 1)
      self.cnn = CNN()
      self.mention_dim += 50
    self.output_dim += self.mention_dim

    if args.model_debug:
      self.mention_proj = nn.Linear(self.mention_dim, 2*args.rnn_dim)
      self.attn = nn.Linear(2*args.rnn_dim, 2*args.rnn_dim)
      self.fusion = Fusion(2*args.rnn_dim)
      self.output_dim = 2*args.rnn_dim*2

    self.batch_num = 0

    if args.add_regu:
      corr_matrix, _, _, mask, mask_inverse = build_concurr_matrix(goal=args.goal)
      corr_matrix -= np.identity(corr_matrix.shape[0])
      self.corr_matrix = torch.from_numpy(corr_matrix).to(torch.device('cuda')).float()
      self.incon_mask = torch.from_numpy(mask).to(torch.device('cuda')).float()
      self.con_mask = torch.from_numpy(mask_inverse).to(torch.device('cuda')).float()

      self.b = nn.Parameter(torch.rand(corr_matrix.shape[0], 1))
      self.b_ = nn.Parameter(torch.rand(corr_matrix.shape[0], 1))

    # Defining LSTM here.   
    self.attentive_sum = SelfAttentiveSum(args.rnn_dim * 2, 100)
    if self.lstm_type == "two":
      self.left_lstm = nn.LSTM(self.embed_dim, 100, bidirectional=True, batch_first=True)
      self.right_lstm = nn.LSTM(self.embed_dim, 100, bidirectional=True, batch_first=True)
    elif self.lstm_type == 'single':
      self.lstm = nn.LSTM(self.embed_dim + 50, args.rnn_dim, bidirectional=True,
                          batch_first=True)
      self.token_mask = nn.Linear(4, 50)

    if args.self_attn:
      self.embed_proj = nn.Linear(self.embed_dim + 50, 2*args.rnn_dim)
      self.encoder = SimpleEncoder(2*args.rnn_dim, head=4, layer=1, dropout=0.2)

    self.loss_func = nn.BCEWithLogitsLoss()
    self.sigmoid_fn = nn.Sigmoid()
    self.goal = args.goal
    self.multitask = args.multitask

    if args.data_setup == 'joint' and args.multitask and args.gcn:
      print("Multi-task learning with gcn on labels")
      self.decoder = GCNMultiDecoder(self.output_dim)
    elif args.data_setup == 'joint' and args.multitask:
      print("Multi-task learning")
      self.decoder = MultiSimpleDecoder(self.output_dim)
    elif args.data_setup == 'joint' and not args.multitask and args.gcn:
      print("Joint training with GCN simple decoder")
      self.decoder = GCNSimpleDecoder(self.output_dim, answer_num, "open"
        )
    elif args.goal == 'onto' and args.gcn:
      print("Ontonotes with gcn decoder")
      self.decoder = GCNSimpleDecoder(self.output_dim, answer_num, "onto")
    else:
      print("Ontonotes using simple decoder")
      self.decoder = SimpleDecoder(self.output_dim, answer_num)

  def sorted_rnn(self, sequences, sequence_lengths, rnn):
    sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(sequences, sequence_lengths)
    packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                 sorted_sequence_lengths.data.tolist(),
                                                 batch_first=True)
    packed_sequence_output, _ = rnn(packed_sequence_input, None)
    unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)
    return unpacked_sequence_tensor.index_select(0, restoration_indices)

  def rnn(self, sequences, lstm):
    outputs, _ = lstm(sequences)
    return outputs.contiguous()

  def define_loss(self, logits, targets, data_type):
    if not self.multitask or data_type == 'onto':
      loss = self.loss_func(logits, targets)
      return loss
    if data_type == 'wiki':
      gen_cutoff, fine_cutoff, final_cutoff = constant.ANSWER_NUM_DICT['gen'], constant.ANSWER_NUM_DICT['kb'], \
                                              constant.ANSWER_NUM_DICT[data_type]
    else:
      gen_cutoff, fine_cutoff, final_cutoff = constant.ANSWER_NUM_DICT['gen'], constant.ANSWER_NUM_DICT['kb'], None
    loss = 0.0
    comparison_tensor = torch.Tensor([1.0]).cuda()
    gen_targets = targets[:, :gen_cutoff]
    fine_targets = targets[:, gen_cutoff:fine_cutoff]
    gen_target_sum = torch.sum(gen_targets, 1)
    fine_target_sum = torch.sum(fine_targets, 1)

    if torch.sum(gen_target_sum.data) > 0:
      gen_mask = torch.squeeze(torch.nonzero(torch.min(gen_target_sum.data, comparison_tensor)), dim=1)
      gen_logit_masked = logits[:, :gen_cutoff][gen_mask, :]
      gen_target_masked = gen_targets.index_select(0, gen_mask)
      gen_loss = self.loss_func(gen_logit_masked, gen_target_masked)
      loss += gen_loss 
    if torch.sum(fine_target_sum.data) > 0:
      fine_mask = torch.squeeze(torch.nonzero(torch.min(fine_target_sum.data, comparison_tensor)), dim=1)
      fine_logit_masked = logits[:,gen_cutoff:fine_cutoff][fine_mask, :]
      fine_target_masked = fine_targets.index_select(0, fine_mask)
      fine_loss = self.loss_func(fine_logit_masked, fine_target_masked)
      loss += fine_loss 

    if not data_type == 'kb':
      if final_cutoff:
        finer_targets = targets[:, fine_cutoff:final_cutoff]
        logit_masked = logits[:, fine_cutoff:final_cutoff]
      else:
        logit_masked = logits[:, fine_cutoff:]
        finer_targets = targets[:, fine_cutoff:]
      if torch.sum(torch.sum(finer_targets, 1).data) >0:
        finer_mask = torch.squeeze(torch.nonzero(torch.min(torch.sum(finer_targets, 1).data, comparison_tensor)), dim=1)
        finer_target_masked = finer_targets.index_select(0, finer_mask)
        logit_masked = logit_masked[finer_mask, :]
        layer_loss = self.loss_func(logit_masked, finer_target_masked)
        loss += layer_loss

    if self.args.add_regu:
      if self.batch_num > self.args.regu_steps:

        # inconsistency loss 1: never concurr, then -1, otherwise log
        # label_matrix = cosine_similarity(self.decoder.linear.weight, self.decoder.linear.weight)
        # target = -1 * self.incon_mask  + self.con_mask * torch.log(self.corr_matrix + 1e-8)
        # auxiliary_loss = ((target - label_matrix) ** 2).mean()
        # loss += self.args.incon_w * auxiliary_loss


        # glove like loss
        less_max_mask = (self.corr_matrix < 100).float()
        greater_max_mask = (self.corr_matrix >= 100).float()
        weight_matrix = less_max_mask * ((self.corr_matrix / 100.0) ** 0.75) + greater_max_mask
        auxiliary_loss = weight_matrix * (torch.mm(self.decoder.linear.weight, self.decoder.linear.weight.t()) + self.b + self.b_.t() - torch.log(self.corr_matrix + 1e-8)) ** 2
        auxiliary_loss = auxiliary_loss.mean()

        # # inconsistency loss 2: only consider these inconsistency labels
        # label_matrix = cosine_similarity(self.decoder.linear.weight, self.decoder.linear.weight)
        # target = -1 * self.incon_mask
        # auxiliary_loss = (((target - label_matrix) * self.incon_mask) ** 2).sum() / self.incon_mask.sum()
        # loss += self.args.incon_w * auxiliary_loss

        # # inconsitenct loss 3: margin loss
        # label_matrix = cosine_similarity(self.decoder.linear.weight, self.decoder.linear.weight)
        # label_consistent = label_matrix * self.con_mask
        # label_contradict = label_matrix * self.incon_mask
        # distance = label_consistent.sum(1) / (self.con_mask.sum(1) + 1e-8) - label_contradict.sum(1) / (self.incon_mask.sum(1) + 1e-8)
        # margin = 0.2
        # auxiliary_loss = torch.max(torch.tensor(0.0).to(torch.device('cuda')), margin - distance).mean()

        loss += self.args.incon_w * auxiliary_loss

    return loss

  def normalize(self, raw_scores, lengths):
      backup = raw_scores.data.clone()
      max_len = raw_scores.size(2)

      for i, length in enumerate(lengths):
          if length == max_len:
              continue
          raw_scores.data[i, :, int(length):] = -1e30

      normalized_scores = F.softmax(raw_scores, dim=-1)
      raw_scores.data.copy_(backup)
      return normalized_scores

  def forward(self, feed_dict, data_type):
    if self.lstm_type == 'two':
      left_outputs = self.rnn(self.input_dropout(feed_dict['left_embed']), self.left_lstm)
      right_outputs = self.rnn(self.input_dropout(feed_dict['right_embed']), self.right_lstm)
      context_rep = torch.cat((left_outputs, right_outputs), 1)
      context_rep, _ = self.attentive_sum(context_rep)
    elif self.lstm_type == 'single':
      token_mask_embed = self.token_mask(feed_dict['token_bio'].view(-1, 4))
      token_mask_embed = token_mask_embed.view(feed_dict['token_embed'].size()[0], -1, 50)
      token_embed = torch.cat((feed_dict['token_embed'], token_mask_embed), 2)
      context_rep_ = self.sorted_rnn(self.input_dropout(token_embed), feed_dict['token_seq_length'], self.lstm)
      if self.args.goal == 'onto' or self.args.model_id == 'baseline':
        context_rep, _ = self.attentive_sum(context_rep_)
      else:
        context_rep, _ = self.attentive_sum(context_rep_, feed_dict["token_seq_length"])

    # Mention Representation
    if self.enhanced_mention:
      if self.args.goal == 'onto'  or self.args.model_id == 'baseline':
        mention_embed, _ = self.head_attentive_sum(feed_dict['mention_embed'])
      else:
        mention_embed, _ = self.head_attentive_sum(feed_dict['mention_embed'], feed_dict['mention_len'])
      span_cnn_embed = self.cnn(feed_dict['span_chars'])
      mention_embed = torch.cat((span_cnn_embed, mention_embed), 1)
    else:
      mention_embed = torch.sum(feed_dict['mention_embed'], dim=1)
    mention_embed = self.mention_dropout(mention_embed)
    # model change
    if self.args.model_debug:
      mention_embed_proj = self.mention_proj(mention_embed).tanh()
      affinity = self.attn(mention_embed_proj.unsqueeze(1)).bmm(F.dropout(context_rep_.transpose(2,1), 0.1, self.training)) # b*1*50
      m_over_c = self.normalize(affinity, feed_dict['token_seq_length'].squeeze().tolist())
      m_retrieve_c = torch.bmm(m_over_c, context_rep_) # b*1*200
      fusioned = self.fusion(m_retrieve_c.squeeze(1), mention_embed_proj)
      output = F.dropout(torch.cat([fusioned, context_rep], dim=1), 0.2, self.training) # seems to be a good choice for ultra-fine
    else:
      output = F.dropout(torch.cat((context_rep, mention_embed), 1), 0.3, self.training)
      # output = torch.cat((context_rep, mention_embed), 1)

    logits = self.decoder(output, data_type)
    loss = self.define_loss(logits, feed_dict['y'], data_type)

    return loss, logits
