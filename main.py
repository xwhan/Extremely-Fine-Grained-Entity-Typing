#!/usr/bin/env python3
import gc
import os
import sys
import time

import torch

import data_utils
import models

from data_utils import to_torch
from eval_metric import mrr
from model_utils import get_eval_string
from model_utils import get_gold_pred_str
from model_utils import get_output_index
from model_utils import metric_dicts
from model_utils import fine_grained_eval
from tensorboardX import SummaryWriter
from torch import optim
import numpy as np
import random
from tqdm import tqdm

sys.path.insert(0, './resources')
import constant

from config_parser import get_logger
from config_parser import read_args

from label_corr import build_concurr_matrix

def get_data_gen(dataname, mode, args, vocab_set, goal, eval_epoch=1):
  dataset = data_utils.TypeDataset(constant.FILE_ROOT + dataname, lstm_type=args.lstm_type,
                                     goal=goal, vocab=vocab_set)
  if mode == 'train':
    data_gen = dataset.get_batch(args.batch_size, args.num_epoch, forever=False, eval_data=False,
                                 simple_mention=not args.enhanced_mention, shuffle=True)
  elif mode == 'dev':
    if args.goal == 'onto':
      eval_batch_size = 2202
    else:
      eval_batch_size = 1998
    data_gen = dataset.get_batch(eval_batch_size, 1, forever=True, eval_data=True,
                                 simple_mention=not args.enhanced_mention)
  else:
    if args.goal == "onto":
      if 'dev' in dataname:
        eval_batch_size = 2202
      else:
        eval_batch_size = 8963
    else:
      eval_batch_size = 1998
      # eval_batch_size = 20
    data_gen = dataset.get_batch(eval_batch_size, eval_epoch, forever=False, eval_data=True,
                                 simple_mention=not args.enhanced_mention)
  return data_gen


def get_joint_datasets(args):
  vocab = data_utils.get_vocab(args.embed_source)
  train_gen_list = []
  valid_gen_list = []
  if args.mode == 'train':
    if not args.remove_open and not args.only_crowd:
      train_gen_list.append(
        #`("open", get_data_gen('train/open*.json', 'train', args, vocab, "open")))
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


def get_datasets(data_lists, args, eval_epoch=1):
  data_gen_list = []
  vocab_set = data_utils.get_vocab(args.embed_source)
  for dataname, mode, goal in data_lists:
    data_gen_list.append(get_data_gen(dataname, mode, args, vocab_set, goal, eval_epoch))
  return data_gen_list

def _train(args):
  logger = get_logger(args)
  if args.data_setup == 'joint':
    train_gen_list, val_gen_list, crowd_dev_gen = get_joint_datasets(args)
  else:
    train_fname = args.train_data
    dev_fname = args.dev_data
    data_gens = get_datasets([(train_fname, 'train', args.goal),
                              (dev_fname, 'dev', args.goal)], args)
    train_gen_list = [(args.goal, data_gens[0])]
    val_gen_list = [(args.goal, data_gens[1])]


  if args.goal == 'onto':
    validation_log = SummaryWriter(os.path.join(constant.EXP_ROOT_ONTO, args.model_id, "log", "validation"))
  else:
    validation_log = SummaryWriter(os.path.join(constant.EXP_ROOT, args.model_id, "log", "validation"))

  model = models.Model(args, constant.ANSWER_NUM_DICT[args.goal])
  model.cuda()
  total_loss = 0
  start_time = time.time()
  init_time = time.time()
  optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
  if args.use_lr_schedule:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [1000], gamma=0.1)

  if args.load:
    load_model(args.reload_model_name, constant.EXP_ROOT, args.model_id, model, optimizer)

  best_f1 = 0
  logger.info('Start training......')
  while True:
    model.batch_num += 1  # single batch composed of all train signal passed by.
    if args.use_lr_schedule:
      scheduler.step()
    for (type_name, data_gen) in train_gen_list:
      try:
        batch = next(data_gen)
        batch, _ = to_torch(batch)
      except StopIteration:
        logger.info(type_name + " finished at " + str(model.batch_num))
        torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   '{0:s}/{1:s}.pt'.format(constant.EXP_ROOT, args.model_id))
        return
      optimizer.zero_grad()
      loss, output_logits = model(batch, type_name)
      loss.backward()
      total_loss += loss.item()
      optimizer.step()

      if model.batch_num % args.log_period == 0 and model.batch_num > 0:
        gc.collect()
        cur_loss = float(1.0 * loss.item())
        elapsed = time.time() - start_time
        train_loss_str = ('|loss {0:3f} | at {1:d}step | @ {2:.2f} ms/batch'.format(cur_loss, model.batch_num,elapsed * 1000 / args.log_period))
        start_time = time.time()
        logger.info(train_loss_str)
    
    if model.batch_num % args.eval_period == 0 and model.batch_num > 0:
      eval_start = time.time()
      logger.info('---- eval at step {0:d} ---'.format(model.batch_num))

      if args.goal == 'onto':
        val_type = "onto"
        feed_dict = next(val_gen_list[0][1])
        EXP_ROOT = constant.EXP_ROOT_ONTO
      else:
        val_type = "open"
        feed_dict = next(crowd_dev_gen)
        EXP_ROOT = constant.EXP_ROOT
      eval_batch, _ = to_torch(feed_dict)
      total_eval_loss, gold_preds = evaluate_batch(model.batch_num, eval_batch, model, val_type, args.goal)
      eval_result, output_str = metric_dicts(gold_preds)
      if args.use_lr_schedule:
        scheduler.step(eval_result['ma_f1'])

      if eval_result['ma_f1'] > 0.78 or args.goal == "open":
        if eval_result['ma_f1'] > best_f1 or model.batch_num > 10000:

          # added for regularization based baselines
          if args.add_regu and model.batch_num < 8000:
            break

          if eval_result['ma_f1'] > best_f1:
            best_f1 = eval_result['ma_f1']
          save_fname = '{0:s}/{1:s}_{2:f}.pt'.format(EXP_ROOT, args.model_id, eval_result['ma_f1'])
          torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_fname)
          logger.critical(
          'Found best. Total {0:.2f} minutes have passed, saving at {1:s} '.format((time.time() - init_time) / 60, save_fname))
        elif args.goal != "open":
          save_fname = '{0:s}/{1:s}_{2:f}.pt'.format(EXP_ROOT, args.model_id, eval_result['ma_f1'])
          torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, save_fname)
          logger.critical(
          'Found best. Total {0:.2f} minutes have passed, saving at {1:s} '.format((time.time() - init_time) / 60, save_fname))

      logger.info('eval loss total: ' + str(total_eval_loss))
      logger.info('eval performance: ' + output_str)
      validation_log.add_scalar('eval_crowd_loss', total_eval_loss, model.batch_num)
      validation_log.add_scalar('eval_crowd_mi_f1', eval_result["f1"], model.batch_num)
      validation_log.add_scalar('eval_crowd_ma_f1', eval_result["ma_f1"], model.batch_num)
      validation_log.add_scalar('eval_crowd_ma_p', eval_result["ma_precision"], model.batch_num)
      validation_log.add_scalar('eval_crowd_ma_recall', eval_result["ma_recall"], model.batch_num)
      logger.info('Eval time clipse {}s'.format(time.time() - eval_start))

      if model.batch_num > args.max_batch:
        break

  # Training finished! 
  torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
             '{0:s}/{1:s}.pt'.format(constant.EXP_ROOT, args.model_id))


def evaluate_batch(batch_num, eval_batch, model, val_type_name, goal):
  model.eval()
  loss, output_logits = model(eval_batch, val_type_name)
  output_index = get_output_index(output_logits)
  # eval_loss = loss.data.cpu().clone()[0]
  eval_loss = loss.item()
  # eval_loss_str = 'Eval loss: {0:.7f} at step {1:d}'.format(eval_loss, model.batch_num)
  gold_pred = get_gold_pred_str(output_index, eval_batch['y'].data.cpu().clone(), goal)
  # eval_accu = sum([set(y) == set(yp) for y, yp in gold_pred]) * 1.0 / len(gold_pred)
  # eval_accus = [set(y) == set(yp) for y, yp in gold_pred]
  # tensorboard.add_validation_scalar('eval_acc_' + val_type_name, eval_accu, model.batch_num)
  # tensorboard.add_validation_scalar('eval_loss_' + val_type_name, eval_loss, model.batch_num)
  # eval_str = get_eval_string(gold_pred)
  # print(val_type_name + ":" +eval_loss_str)
  # print(gold_pred[:3])
  # print(val_type_name+":"+ eval_str)
  # logging.info(val_type_name + ":" + eval_loss_str)
  # logging.info(val_type_name +":" +  eval_str)
  model.train()
  return eval_loss, gold_pred


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
    print(param_str)
  print('Loading model from ... {0:s}'.format(model_file_name))


def visualize(args):
  saved_path = constant.EXP_ROOT
  model = models.Model(args, constant.ANSWER_NUM_DICT[args.goal])
  model.cuda()
  model.eval()
  model.load_state_dict(torch.load(saved_path + '/' + args.model_id + '_best.pt')["state_dict"])

  label2id = constant.ANS2ID_DICT["open"] 
  visualize = SummaryWriter("../visualize/" + args.model_id)
  # label_list = ["person", "leader", "president", "politician", "organization", "company", "athlete","adult",  "male",  "man", "television_program", "event"]
  label_list = list(label2id.keys())
  ids = [label2id[_] for _ in label_list]
  if args.gcn:
    # connection_matrix = model.decoder.label_matrix + model.decoder.weight * model.decoder.affinity
    connection_matrix = model.decoder.label_matrix + model.decoder.weight * model.decoder.affinity
    label_vectors = model.decoder.transform(connection_matrix.mm(model.decoder.linear.weight) / connection_matrix.sum(1, keepdim=True))
  else:
    label_vectors = model.decoder.linear.weight.data

  interested_vectors = torch.index_select(label_vectors, 0, torch.tensor(ids).to(torch.device("cuda")))
  visualize.add_embedding(interested_vectors, metadata=label_list, label_img=None)

def _test(args):
  assert args.load
  test_fname = args.eval_data
  model = models.Model(args, constant.ANSWER_NUM_DICT[args.goal])
  model.cuda()
  model.eval()
  # load_model(args.reload_model_name, constant.EXP_ROOT, args.model_id, model)

  if args.goal == "onto":
    saved_path = constant.EXP_ROOT_ONTO
  else:
    saved_path = constant.EXP_ROOT
  model.load_state_dict(torch.load(saved_path + '/' + args.model_id + '_best.pt')["state_dict"])
  
  data_gens = get_datasets([(test_fname, 'test', args.goal)], args, eval_epoch=1)
  for name, dataset in [(test_fname, data_gens[0])]:
    print('Processing... ' + name)

    batch = next(dataset)
    eval_batch, _ = to_torch(batch)
    loss, output_logits = model(eval_batch, args.goal)

    threshes = np.arange(0,1,0.02)
    # threshes = [0.65, 0.68, 0.7, 0.71]
    # threshes = [0.5]
    p_and_r = []
    for thresh in tqdm(threshes):
      total_gold_pred = []
      total_probs = []
      total_ys = []
      print('\nthresh {}'.format(thresh))
      output_index = get_output_index(output_logits, thresh)
      output_prob = model.sigmoid_fn(output_logits).data.cpu().clone().numpy()
      y = eval_batch['y'].data.cpu().clone().numpy()
      gold_pred = get_gold_pred_str(output_index, y, args.goal)

      total_probs.extend(output_prob)
      total_ys.extend(y)
      total_gold_pred.extend(gold_pred)
      # mrr_val = mrr(total_probs, total_ys)

      # json.dump(gold_pred, open('nomulti_predictions.json', 'w'))
      # np.save('y', total_ys)
      # np.save('probs', total_probs)

      # print('mrr_value: ', mrr_val)
      # result, eval_str = metric_dicts(total_gold_pred)
      result, eval_str = fine_grained_eval(total_gold_pred)

      # fine_grained_eval(total_gold_pred)

      p_and_r.append([result["ma_precision"], result["ma_recall"]])
      print(eval_str)

    np.save(saved_path + '/{}_pr_else_dev'.format(args.model_id), p_and_r)


if __name__ == '__main__':
  config = read_args()

  # fix random seed
  np.random.seed(config.seed)
  random.seed(config.seed)
  torch.cuda.manual_seed(config.seed)
  torch.cuda.manual_seed_all(config.seed)

  if config.mode == 'train':
    _train(config)
  elif config.mode == 'test':
    _test(config)
  elif config.mode == 'visual':
    visualize(config)
  else:
    raise ValueError("invalid value for 'mode': {}".format(config.mode))
