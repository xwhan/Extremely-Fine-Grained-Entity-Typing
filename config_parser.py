import argparse
import logging

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="Identifier for model")
    # Data
    parser.add_argument("-train_data", help="Train data", default="ontonotes/g_train.json")
    parser.add_argument("-dev_data", help="Dev data", default="ontonotes/g_dev.json")
    parser.add_argument("-eval_data", help="Test data", default="ontonotes/g_test.json")
    # parser.add_argument("-num_epoch", help="The number of epoch", default=5000, type=int)
    parser.add_argument("-num_epoch", help="The number of epoch", default=5000, type=int)
    parser.add_argument("-batch_size", help="The batch size", default=1000, type=int)
    parser.add_argument("-eval_batch_size", help="The batch size", default=1998, type=int)
    parser.add_argument("-goal", help="Limiting vocab to smaller vocabs (either ontonote or figer)", default="open",
                        choices=["open", "onto", "wiki", 'kb'])
    parser.add_argument("-seed", help="Pytorch random Seed", default=1888, type=int)
    parser.add_argument("-gpu", help="Using gpu or cpu", default=False, action="store_true")

    parser.add_argument("-embed_source", default='glove', type=str)
    parser.add_argument("-max_batch", default=50000, type=int)

    # learning
    parser.add_argument("-mode", help="Whether to train or test", default="train", choices=["train", "test", "visual"])
    parser.add_argument("-learning_rate", help="start learning rate", default=0.001, type=float)
    parser.add_argument("-mention_dropout", help="drop out rate for mention", default=0.5, type=float)
    parser.add_argument("-input_dropout", help="drop out rate for sentence", default=0.2, type=float)
    parser.add_argument("-incon_w", default=0.2, type=float)
    parser.add_argument("-use_lr_schedule", action='store_true')
    parser.add_argument("-use_sparse_adam", action='store_true')

    # Data ablation study
    parser.add_argument("-add_crowd", help="Add indomain data as train", default=False, action='store_true')
    parser.add_argument("-data_setup", help="Whether to use joint data set-up", default="single",
                        choices=["single", "joint"])
    parser.add_argument("-only_crowd", help="Only using indomain data as train", default=False, action='store_true')
    parser.add_argument("-remove_el", help="Remove supervision from entity linking", default=False, action='store_true')
    parser.add_argument("-remove_open", help="Remove supervision from headwords", default=False, action='store_true')

    # Model
    parser.add_argument("-multitask", help="Using a multitask loss term.", default=False, action='store_true')
    parser.add_argument("-enhanced_mention", help="Use attention and cnn for mention representation", default=False, action='store_true')
    parser.add_argument("-lstm_type", default="two", choices=["two", "single"])
    parser.add_argument("-dim_hidden", help="The number of hidden dimension.", default=100, type=int)
    parser.add_argument("-rnn_dim", help="The number of RNN dimension.", default=100, type=int)
    # Save / log related
    # parser.add_argument("-save_period", help="How often to save", default=5000, type=int)
    parser.add_argument("-eval_period", help="How often to run dev", default=1000, type=int)
    parser.add_argument("-log_period", help="How often to save", default=1000, type=int)

    parser.add_argument("-load", help="Load existing model.", action='store_true')
    parser.add_argument("-reload_model_name", help="")

    # debugging model architextures
    parser.add_argument("-model_debug", action='store_true')
    parser.add_argument("-gcn", action='store_true', help='whether to use')
    parser.add_argument("-add_regu", action='store_true')
    parser.add_argument("-regu_steps", default=8000, type=int)
    parser.add_argument("-self_attn", action='store_true', help="replace LSTM with self-attention encoder")
    parser.add_argument("-label_prop", action='store_true')
    parser.add_argument("-thresh", default=0.5, type=float)

    args = parser.parse_args()

    if args.goal == 'onto':
        args.eval_period = 10

    return args

def get_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%m-%d %H:%M:%S')
    fh = logging.FileHandler('./logs/{}.txt'.format(args.model_id), mode='w+')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info("------HYPERPARAMETERS-------")
    for k, v in vars(args).items():
        logger.info(k + ': ' + str(v))
    logger.info("----------------------------")

    return logger
