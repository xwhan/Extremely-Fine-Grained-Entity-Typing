Code for our NAACL 2019 paper:

## Imposing Label-Relational Inductive Bias for Extremely Fine-Grained Entity Typing

Paper link: [http://arxiv.org/abs/1903.02591](http://arxiv.org/abs/1903.02591)

Model Overview:
<p align="center"><img width="85%" src="imgs/main.png" /></p>

### Requirements
* ``PyTorch 0.4.1``
* ``tensorboardX``
* ``tqdm``
* ``gluonnlp``

### Running the code
First prepare the dataset and embeddings
* download data from [http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz](http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz), unzip if and put it under ``data/``


#### 1. Ultra-Fine experiments (10331 free-text labels and millions of training data)

##### Train the best model on Ultra-Fine 
``CUDA_VISIBLE_DEVICES=1 python main.py $RUN_ID$ -lstm_type single -model_debug -enhanced_mention -data_setup joint -add_crowd -multitask -gcn``

##### You can then test your saved model
``CUDA_VISIBLE_DEVICES=1 python main.py $RUN_ID$ -lstm_type single -model_debug -enhanced_mention -data_setup joint -add_crowd -multitask -gcn -load -mode test -eval_data crowd/test.json``


##### Ablation experiments
**a) w/o gcn**

``CUDA_VISIBLE_DEVICES=1 python main.py $RUN_ID$ -lstm_type single -model_debug -enhanced_mention -data_setup joint -add_crowd -multitask``

**b) w/o enhanced mention-context interaction**

``CUDA_VISIBLE_DEVICES=1 python main.py $RUN_ID$ -lstm_type single -gcn -enhanced_mention -data_setup joint -add_crowd -multitask ``


#### 2. Experiments on OntoNotes
**Training**

``CUDA_VISIBLE_DEVICES=1 python main.py $RUN_ID$ -lstm_type single -enhanced_mention -goal onto -gcn``

**Testing**

``CUDA_VISIBLE_DEVICES=1 python main.py $RUN_ID$ -lstm_type single -enhanced_mention -goal onto -gcn -mode test -load -eval_data ontonotes/g_dev.json``

#### Notes
**The meaning of the arguments can be found in ``config_parser.py``**

### Acknowledgement
We thank [Choi et al](https://homes.cs.washington.edu/~eunsol/papers/acl_18.pdf) for the release of the Ultra-Fine dataset and the basic model: [https://github.com/uwnlp/open_type](https://github.com/uwnlp/open_type).