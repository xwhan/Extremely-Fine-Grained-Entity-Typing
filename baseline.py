#!/usr/bin/env python3
import datetime
import gc
import logging
import pickle
import os
import sys
import time, json

import torch

import data_utils
import models
from data_utils import to_torch
from eval_metric import mrr
from model_utils import get_gold_pred_str, get_eval_string, get_output_index
from tensorboardX import SummaryWriter
from torch import optim
from tqdm import tqdm
import numpy as np

sys.path.insert(0, './resources')
import config_parser, constant, eval_metric


class TensorboardWriter:
  """
  Wraps a pair of ``SummaryWriter`` instances but is a no-op if they're ``None``.
  Allows Tensorboard logging without always checking for Nones first.
  """
  def __init__(self, train_log: SummaryWriter = None, validation_log: SummaryWriter = None) -> None:
    self._train_log = train_log
    self._validation_log = validation_log

  def add_train_scalar(self, name: str, value: float, global_step: int) -> None:
    if self._train_log is not None:
      self._train_log.add_scalar(name, value, global_step)

  def add_validation_scalar(self, name: str, value: float, global_step: int) -> None:
    if self._validation_log is not None:
      self._validation_log.add_scalar(name, value, global_step)


def get_data_gen(dataname, mode, args, vocab_set, goal):
  dataset = data_utils.TypeDataset(constant.FILE_ROOT + dataname, lstm_type=args.lstm_type,
                                     goal=goal, vocab=vocab_set)
  if mode == 'train':
    data_gen = dataset.get_batch(args.batch_size, args.num_epoch, forever=False, eval_data=False,
                                 simple_mention=not args.enhanced_mention)
  elif mode == 'dev':
    data_gen = dataset.get_batch(args.eval_batch_size, 1, forever=True, eval_data=True,
                                 simple_mention=not args.enhanced_mention)
  else:
    data_gen = dataset.get_batch(args.eval_batch_size, 1, forever=False, eval_data=True,
                                 simple_mention=not args.enhanced_mention)
  return data_gen


def get_joint_datasets(args):
  vocab = data_utils.get_vocab()
  train_gen_list = []
  valid_gen_list = []
  if args.mode == 'train':
    if not args.remove_open and not args.only_crowd:
      train_gen_list.append(
        #("open", get_data_gen('train/open*.json', 'train', args, vocab, "open")))
        ("open", get_data_gen('distant_supervision/headword_train.json', 'train', args, vocab, "open")))
      valid_gen_list.append(("open", get_data_gen('distant_supervision/headword_dev.json', 'dev', args, vocab, "open")))
    if not args.remove_el and not args.only_crowd:
      valid_gen_list.append(
        ("wiki",
         get_data_gen('distant_supervision/el_dev.json', 'dev', args, vocab, "wiki" if args.multitask else "open")))
      train_gen_list.append(
        ("wiki",
         get_data_gen('distant_supervision/el_train.json', 'train', args, vocab, "wiki" if args.multitask else "open")))
         #get_data_gen('train/el_train.json', 'train', args, vocab, "wiki" if args.multitask else "open")))
    if args.add_crowd or args.only_crowd:
      train_gen_list.append(
        ("open", get_data_gen('crowd/train_m.json', 'train', args, vocab, "open")))
  crowd_dev_gen = get_data_gen('crowd/dev.json', 'dev', args, vocab, "open")
  return train_gen_list, valid_gen_list, crowd_dev_gen


def get_datasets(data_lists, args):
  data_gen_list = []
  vocab_set = data_utils.get_vocab()
  for dataname, mode, goal in data_lists:
    data_gen_list.append(get_data_gen(dataname, mode, args, vocab_set, goal))
  return data_gen_list


def _train(args):
  if args.data_setup == 'joint':
    train_gen_list, val_gen_list, crowd_dev_gen = get_joint_datasets(args)
  else:
    train_fname = args.train_data
    dev_fname = args.dev_data
    data_gens = get_datasets([(train_fname, 'train', args.goal),
                              (dev_fname, 'dev', args.goal)], args)
    train_gen_list = [(args.goal, data_gens[0])]
    val_gen_list = [(args.goal, data_gens[1])]
  train_log = SummaryWriter(os.path.join(constant.EXP_ROOT, args.model_id, "log", "train"))
  validation_log = SummaryWriter(os.path.join(constant.EXP_ROOT, args.model_id, "log", "validation"))
  tensorboard = TensorboardWriter(train_log, validation_log)

  model = models.Model(args, constant.ANSWER_NUM_DICT[args.goal])
  model.cuda()
  total_loss = 0
  batch_num = 0
  start_time = time.time()
  init_time = time.time()
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

  if args.load:
    load_model(args.reload_model_name, constant.EXP_ROOT, args.model_id, model, optimizer)

  for idx, m in enumerate(model.modules()):
    logging.info(str(idx) + '->' + str(m))

  best_eval_ma_f1=0
  while True:
    batch_num += 1  # single batch composed of all train signal passed by.
    for (type_name, data_gen) in train_gen_list:
      try:
        batch = next(data_gen)
        batch, _ = to_torch(batch)
      except StopIteration:
        logging.info(type_name + " finished at " + str(batch_num))
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   '{0:s}/{1:s}.pt'.format(constant.EXP_ROOT, args.model_id))
        return
      optimizer.zero_grad()
      loss, output_logits = model(batch, type_name)
      loss.backward()
      total_loss += loss.data.cpu()[0]
      optimizer.step()

#      if batch_num % args.log_period == 0 and batch_num > 0:
#        gc.collect()
#        cur_loss = float(1.0 * loss.data.cpu().clone()[0])
#        elapsed = time.time() - start_time
#        train_loss_str = ('|loss {0:3f} | at {1:d}step | @ {2:.2f} ms/batch'.format(cur_loss, batch_num,
#                                                                                    elapsed * 1000 / args.log_period))
#        start_time = time.time()
#        print(train_loss_str)
#        logging.info(train_loss_str)
#        tensorboard.add_train_scalar('train_loss_' + type_name, cur_loss, batch_num)
#
#       if batch_num % args.eval_period == 0 and batch_num > 0:
#         output_index = get_output_index(output_logits)
#         gold_pred_train = get_gold_pred_str(output_index, batch['y'].data.cpu().clone(), args.goal)
#         accuracy = sum([set(y) == set(yp) for y, yp in gold_pred_train]) * 1.0 / len(gold_pred_train)
#         train_acc_str = '{1:s} Train accuracy: {0:.1f}%'.format(accuracy * 100, type_name)
#         print(train_acc_str)
#         logging.info(train_acc_str)
#         tensorboard.add_train_scalar('train_acc_' + type_name, accuracy, batch_num)
#         for (val_type_name, val_data_gen) in val_gen_list:
#           if val_type_name == type_name:
#             eval_batch, _ = to_torch(next(val_data_gen))
#             evaluate_batch(batch_num, eval_batch, model, tensorboard, val_type_name, args.goal)

    if batch_num % args.eval_period == 0 and batch_num > 0:
      # Evaluate Loss on the Turk Dev dataset.
      print('---- eval at step {0:d} ---'.format(batch_num))
      feed_dict = next(crowd_dev_gen)
      eval_batch, _ = to_torch(feed_dict)
      crowd_eval_loss, crowd_eval_ma_f1 = evaluate_batch(batch_num, eval_batch, model, tensorboard, "open", "open")

    if batch_num % args.save_period == 0 and batch_num > 0 and crowd_eval_ma_f1 > best_eval_ma_f1:
      best_eval_ma_f1 = crowd_eval_ma_f1
      save_fname = '{0:s}/{1:s}_best.pt'.format(constant.EXP_ROOT, args.model_id)
      torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_fname)
      print(
        'Total {0:.2f} minutes have passed, saving at {1:s} '.format((time.time() - init_time) / 60, save_fname))
  # Training finished!
  torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
             '{0:s}/{1:s}.pt'.format(constant.EXP_ROOT, args.model_id))


def evaluate_batch(batch_num, eval_batch, model, tensorboard, val_type_name, goal):
  model.eval()
  loss, output_logits = model(eval_batch, val_type_name)
  output_index = get_output_index(output_logits)
  eval_loss = loss.data.cpu().clone()[0]
  eval_loss_str = 'Eval loss: {0:.7f} at step {1:d}'.format(eval_loss, batch_num)
  gold_pred = get_gold_pred_str(output_index, eval_batch['y'].data.cpu().clone(), goal)
  eval_accu = sum([set(y) == set(yp) for y, yp in gold_pred]) * 1.0 / len(gold_pred)
  tensorboard.add_validation_scalar('eval_acc_' + val_type_name, eval_accu, batch_num)
  tensorboard.add_validation_scalar('eval_loss_' + val_type_name, eval_loss, batch_num)
  eval_str, ma_f1, f1 = get_eval_string(gold_pred)
  print(val_type_name + ":" +eval_loss_str)
  print(gold_pred[:3])
  print(val_type_name+":"+ eval_str)
  logging.info(val_type_name + ":" + eval_loss_str)
  logging.info(val_type_name +":" +  eval_str)
  model.train()
  tensorboard.add_validation_scalar('ma_f1' + val_type_name, ma_f1, batch_num)
  tensorboard.add_validation_scalar('f1' + val_type_name, f1, batch_num)
  return eval_loss, ma_f1


def load_model(reload_model_name, save_dir, model_id, model, optimizer=None):
  if reload_model_name:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, reload_model_name)
  else:
    model_file_name = '{0:s}/{1:s}.pt'.format(save_dir, model_id)
  checkpoint = torch.load(model_file_name)
  model.load_state_dict(checkpoint['state_dict'])
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer'])
  else:
    total_params = 0
    # Log params
    for k in checkpoint['state_dict']:
      elem = checkpoint['state_dict'][k]
      param_s = 1
      for size_dim in elem.size():
        param_s = size_dim * param_s
      print(k, elem.size())
      total_params += param_s
    param_str = ('Number of total parameters..{0:d}'.format(total_params))
    logging.info(param_str)
    print(param_str)
  logging.info("Loading old file from {0:s}".format(model_file_name))
  print('Loading model from ... {0:s}'.format(model_file_name))


def _test(args):
  assert args.load
  test_fname = args.eval_data
  data_gens = get_datasets([(test_fname, 'test', args.goal)], args)
  model = models.Model(args, constant.ANSWER_NUM_DICT[args.goal])
  model.cuda()
  model.eval()
  # load_model(args.reload_model_name, constant.EXP_ROOT, args.model_id, model)

  saved_path = constant.EXP_ROOT
  model.load_state_dict(torch.load(saved_path + '/' + args.model_id + '_best.pt')["state_dict"])
  data_gens = get_datasets([(test_fname, 'test', args.goal)], args)#, eval_epoch=1)
  for name, dataset in [(test_fname, data_gens[0])]:
    print('Processing... ' + name)
    batch = next(dataset)
    eval_batch, annot_ids = to_torch(batch)
    loss, output_logits = model(eval_batch, args.goal)

    threshes = np.arange(0,1,0.005)
    p_and_r = []
    for thresh in tqdm(threshes):
      total_gold_pred = []
      total_annot_ids = []
      total_probs = []
      total_ys = []
      print('thresh {}'.format(thresh))
      output_index = get_output_index(output_logits, thresh)
      output_prob = model.sigmoid_fn(output_logits).data.cpu().clone().numpy()
      y = eval_batch['y'].data.cpu().clone().numpy()
      gold_pred = get_gold_pred_str(output_index, y, args.goal)
      total_probs.extend(output_prob)
      total_ys.extend(y)
      total_gold_pred.extend(gold_pred)
      total_annot_ids.extend(annot_ids)
      # mrr_val = mrr(total_probs, total_ys)
      # print('mrr_value: ', mrr_val)
      # pickle.dump({'gold_id_array': total_ys, 'pred_dist': total_probs},
                  # open('./{0:s}.p'.format(args.reload_model_name), "wb"))
      # with open('./{0:s}.json'.format(args.reload_model_name), 'w') as f_out:
      #   output_dict = {}
      #   for a_id, (gold, pred) in zip(total_annot_ids, total_gold_pred):
      #     output_dict[a_id] = {"gold": gold, "pred": pred}
      #   json.dump(output_dict, f_out)
      eval_str, p, r = get_eval_string(total_gold_pred)
      p_and_r.append([p, r])
      print(eval_str)

    np.save(saved_path + '/baseline_pr_dev', p_and_r)

  # for name, dataset in [(test_fname, data_gens[0])]:
  #   print('Processing... ' + name)
  #   total_gold_pred = []
  #   total_annot_ids = []
  #   total_probs = []
  #   total_ys = []
  #   for batch_num, batch in enumerate(dataset):
  #     eval_batch, annot_ids = to_torch(batch)
  #     loss, output_logits = model(eval_batch, args.goal)
  #     output_index = get_output_index(output_logits)
  #     output_prob = model.sigmoid_fn(output_logits).data.cpu().clone().numpy()
  #     y = eval_batch['y'].data.cpu().clone().numpy()
  #     gold_pred = get_gold_pred_str(output_index, y, args.goal)
  #     total_probs.extend(output_prob)
  #     total_ys.extend(y)
  #     total_gold_pred.extend(gold_pred)
  #     total_annot_ids.extend(annot_ids)
  #   mrr_val = mrr(total_probs, total_ys)
  #   print('mrr_value: ', mrr_val)
  #   pickle.dump({'gold_id_array': total_ys, 'pred_dist': total_probs},
  #               open('./{0:s}.p'.format(args.reload_model_name), "wb"))
  #   with open('./{0:s}.json'.format(args.reload_model_name), 'w') as f_out:
  #     output_dict = {}
  #     for a_id, (gold, pred) in zip(total_annot_ids, total_gold_pred):
  #       output_dict[a_id] = {"gold": gold, "pred": pred}
  #     json.dump(output_dict, f_out)
  #   eval_str = get_eval_string(total_gold_pred)
  #   print(eval_str)
  #   logging.info('processing: ' + name)
  #   logging.info(eval_str)

if __name__ == '__main__':
  config = config_parser.parser.parse_args()
  torch.cuda.manual_seed(config.seed)
  logging.basicConfig(
    filename=constant.EXP_ROOT +"/"+ config.model_id + datetime.datetime.now().strftime("_%m-%d_%H") + config.mode + '.txt',
    level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
  logging.info(config)
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if config.mode == 'train':
    _train(config)
  elif config.mode == 'test':
    _test(config)
  else:
    raise ValueError("invalid value for 'mode': {}".format(config.mode))
