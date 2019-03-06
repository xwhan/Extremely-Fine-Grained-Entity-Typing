import numpy as np
import json, sys, pickle
from eval_metric import mrr, macro

def stratify(all_labels, types):
  """
  Divide label into three categories.
  """
  coarse = types[:9]
  fine = types[9:130]
  return ([l for l in all_labels if l in coarse],
          [l for l in all_labels if ((l in fine) and (not l in coarse))],
          [l for l in all_labels if (not l in coarse) and (not l in fine)])

def get_mrr(pred_fname):
  dicts = pickle.load(open(pred_fname, "rb"))
  mrr_value = mrr(dicts['pred_dist'], dicts['gold_id_array'])
  return mrr_value

def compute_prf1(fname):
  with open(fname) as f:
    total = json.load(f)  
  true_and_predictions = []
  for k, v in total.items():
    true_and_predictions.append((v['gold'], v['pred']))
  count, pred_count, avg_pred_count, p, r, f1 = macro(true_and_predictions)
  perf_total = "{0}\t{1:.2f}\tP:{2:.1f}\tR:{3:.1f}\tF1:{4:.1f}".format(count, avg_pred_count, p * 100,
                                                                    r * 100, f1 * 100)
  print(perf_total)

def compute_granul_prf1(fname, type_fname):
  with open(fname) as f:
    total = json.load(f)  
  coarse_true_and_predictions = []
  fine_true_and_predictions = []
  finer_true_and_predictions = []
  with open(type_fname) as f:
    types = [x.strip() for x in f.readlines()]
  for k, v in total.items():
    coarse_gold, fine_gold, finer_gold = stratify(v['gold'], types)
    coarse_pred, fine_pred, finer_pred = stratify(v['pred'], types)
    coarse_true_and_predictions.append((coarse_gold, coarse_pred))
    fine_true_and_predictions.append((fine_gold, fine_pred))
    finer_true_and_predictions.append((finer_gold, finer_pred))

  for true_and_predictions in [coarse_true_and_predictions, fine_true_and_predictions, finer_true_and_predictions]:
    count, pred_count, avg_pred_count, p, r, f1 = macro(true_and_predictions)
    perf = "{0}\t{1:.2f}\tP:{2:.1f}\tR:{3:.1f}\tF1:{4:.1f}".format(count, avg_pred_count, p * 100,
                                                                    r * 100, f1 * 100)
    print(perf)

def load_augmented_input(fname):
  output_dict = {}
  with open(fname) as f:
    for line in f:
      elem = json.loads(line.strip())
      mention_id = elem.pop("annot_id")
      output_dict[mention_id] = elem
  return output_dict

def visualize(gold_pred_fname, original_fname, type_fname):
  with open(gold_pred_fname) as f:
    total = json.load(f) 
  original = load_augmented_input(original_fname)
  with open(type_fname) as f:
    types = [x.strip() for x in f.readlines()]
  for annot_id, v in total.items():
    elem = original[annot_id]
    mention = elem['mention_span']
    left = elem['left_context_token']
    right = elem['right_context_token']
    text_str = ' '.join(left)+" __"+mention+"__ "+' '.join(right)
    gold = v['gold']
    print('  |  '.join([text_str, ', '.join([("__"+v+"__" if v in gold else v )for v in v['pred']]), ','.join(gold)]))

if __name__ == '__main__':
  gold_pred_str_fname = sys.argv[1]+'.json'
  mrr_fname = sys.argv[1]+'.p'
  type_fname = './resources/types.txt'
  # compute mrr
  mrr_value = get_mrr(mrr_fname)
  print("MRR {0:.4f}".format(mrr_value))
  # compute precision, recall, f1
  compute_prf1(gold_pred_str_fname)
  print('printing performance for coarse, fine, finer labels in order')
  compute_granul_prf1(gold_pred_str_fname, type_fname)
