from fastNLP.io.pipe.conll import OntoNotesNERPipe, Conll2003NERPipe
from fastNLP import SpanFPreRecMetric, BucketSampler
from fastNLP.embeddings import StaticEmbedding, StackEmbedding, LSTMCharEmbedding, ElmoEmbedding, BertEmbedding, CNNCharEmbedding
from fastNLP.models import BiLSTMCRF
from fastNLP import SpanFPreRecMetric
from torch.optim import Adam
from fastNLP import LossInForward
from fastNLP import Trainer
import torch
from fastNLP import Tester
from modules.utils import EvaluateNER
from modules.callbacks import EvaluateCallback
from modules.TransformerEmbedding import TransformerCharEmbed
import argparse
import sys
import os

parser = argparse.ArgumentParser()


parser.add_argument('--dataset', type=str, default='conll2003',
                    choices=['conll2003', 'en-ontonotes', 'drone', 'drone-polysemi'])
parser.add_argument('--char_embed', type=str, choices=[
                    'lstm', 'cnn', 'adatrans'], default=None, help='Char Embedding methods')
parser.add_argument('--word_embed', type=str, choices=['glove', 'bert', 'elmo',
                    'word2vec', 'fasttext'], default='static', help='Type of Word Embdding used')
parser.add_argument('--output_dir', type=str,
                    help="Where to store the output files")
parser.add_argument('--counter', type=int, help='Counter of loop')

args = parser.parse_args()
dataset = args.dataset
word_type = args.word_embed
char_type = args.char_embed
normalize_embed = True
scale = False
output_dir = os.path.join(args.output_dir, '{}'.format(dataset))
encoding_type = 'bioes'
n_epochs = 50
pos_embed = None
counter = args.counter
attn_type = 'bilstm'
lr = 1e-2
batch_size = 8
# sys.stdout = open(
#     "./output-2/{}_{}_{}.txt".format('BiLSTM-CRF', word_type, char_type), "w")


def load_data():
    # 替换路径
    if dataset == 'conll2003':
        # conll2003的lr不能超过0.002
        paths = {'test': "../data/conll2003/test.txt",
                 'train': "../data/conll2003/train.txt",
                 'dev': "../data/conll2003/dev.txt"}
        data = Conll2003NERPipe(
            encoding_type=encoding_type).process_from_file(paths)
    elif dataset == 'en-ontonotes':
        # 会使用这个文件夹下的train.txt, test.txt, dev.txt等文件
        paths = '../data/en-ontonotes/english'
        data = OntoNotesNERPipe(
            encoding_type=encoding_type).process_from_file(paths)
    elif dataset == 'drone':
        paths = {
            'train': "./data/drone/train.txt",
            'dev': "./data/drone/test.txt",
            'test': "./data/drone/test.txt",
        }
        data = Conll2003NERPipe(
            encoding_type=encoding_type).process_from_file(paths)
    elif dataset == 'drone-polysemi':
        paths = {
            'train': "./data/drone-polysemi/train.txt",
            'dev': "./data/drone-polysemi/test.txt",
            'test': "./data/drone-polysemi/test.txt",
        }
        data = Conll2003NERPipe(
            encoding_type=encoding_type).process_from_file(paths)

    char_embed = None
    if char_type == 'cnn':
        char_embed = CNNCharEmbedding(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, filter_nums=[30],
                                      kernel_sizes=[3], word_dropout=0, dropout=0.3, pool_method='max', include_word_start_end=False, min_char_freq=2)
    elif char_type in ['adatrans', 'naive']:
        char_embed = TransformerCharEmbed(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, word_dropout=0,
                                          dropout=0.3, pool_method='max', activation='relu',
                                          min_char_freq=2, requires_grad=True, include_word_start_end=False,
                                          char_attn_type=char_type, char_n_head=3, char_dim_ffn=60, char_scale=scale,
                                          char_dropout=0.15, char_after_norm=True)
    elif char_type == 'lstm':
        char_embed = LSTMCharEmbedding(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, word_dropout=0,
                                       dropout=0.3, hidden_size=100, pool_method='max', activation='relu',
                                       min_char_freq=2, bidirectional=True, requires_grad=True, include_word_start_end=False)

    if word_type == 'bert':
        word_embed = BertEmbedding(vocab=data.get_vocab('words'),
                                   model_dir_or_name='en-base-cased',
                                   requires_grad=False, lower=False,
                                   word_dropout=0, dropout=0.5,
                                   layers='4,-2,-1')
    elif word_type == 'elmo':
        word_embed = ElmoEmbedding(vocab=data.get_vocab('words'), model_dir_or_name='en-original', layers='mix', requires_grad=True,
                                   word_dropout=0.0, dropout=0.5, cache_word_reprs=False)
        word_embed.set_mix_weights_requires_grad()
    elif word_type == 'glove':
        word_embed = StaticEmbedding(vocab=data.get_vocab('words'),
                                     model_dir_or_name='en-glove-6b-100d',
                                     requires_grad=True, lower=False, word_dropout=0, dropout=0.5,
                                     only_norm_found_vector=normalize_embed)
    elif word_type == 'word2vec':
        word_embed = StaticEmbedding(vocab=data.get_vocab('words'),
                                     model_dir_or_name='en-word2vec-300d',
                                     requires_grad=True, lower=False, word_dropout=0, dropout=0.5,
                                     only_norm_found_vector=normalize_embed)
    elif word_type == 'fasttext':
        word_embed = StaticEmbedding(vocab=data.get_vocab('words'),
                                     model_dir_or_name='en-fasttext-crawl',
                                     requires_grad=True, lower=False, word_dropout=0, dropout=0.5,
                                     only_norm_found_vector=normalize_embed)
    if char_embed is not None:
        embed = StackEmbedding([word_embed, char_embed],
                               dropout=0, word_dropout=0.02)
    else:
        word_embed.word_drop = 0.02
        embed = word_embed

    data.rename_field('chars', 'words')
    return data, embed


data_bundle, embed = load_data()
print(data_bundle)

model = BiLSTMCRF(embed=embed, num_classes=len(data_bundle.get_vocab('target')), num_layers=1, hidden_size=200, dropout=0.5,
                  target_vocab=data_bundle.get_vocab('target'))

metric = SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab(
    'target'), encoding_type=encoding_type, f_type='micro', only_gross=True)
# metric = EvaluateNER()
optimizer = Adam(model.parameters(), lr=lr)
loss = LossInForward()
device = 0
evaluate_callback = EvaluateCallback(data_bundle.get_dataset('test'))
trainer = Trainer(data_bundle.get_dataset('train'), model, loss=loss, optimizer=optimizer, batch_size=batch_size,
                  dev_data=data_bundle.get_dataset('dev'), n_epochs=n_epochs, metrics=metric, callbacks=[evaluate_callback], dev_batch_size=batch_size, device=device, test_use_tqdm=False,
                  use_tqdm=True, print_every=300)

output_folder = os.path.join(
    'output-2', '{}'.format(dataset), '{}'.format(attn_type))
if (not os.path.exists(output_folder)):
    os.makedirs(output_folder)

sys.stdout = open("./output-2/{}/{}/{}-{}_{}_{}_{}_{}.txt".format(dataset, attn_type, 'scaled' if scale ==
                  True else 'unscaled', attn_type, word_type, char_type, pos_embed, counter), "w")

# Print model's config
print("Hyperparams:\n")
print("dataset: ", dataset)
print("n_heads: ", 0)
print("head_dims: ", 0)
print("num_layers: ", 0)
print("lr: ", lr)
print("encoding_type: ", encoding_type)
print("pos_embed: ", pos_embed)
print("char_type: ", char_type)
print("word_type: ", word_type)
print("attn_type: ", attn_type)
print("n_epochs: ", n_epochs)
print("batch_size: {}\n".format(batch_size))

trainer.train(load_best_model=True)
tester = Tester(data_bundle.get_dataset('test'), model, metrics=SpanFPreRecMetric(
    tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type, f_type='micro', only_gross=False))
tester.test()
sys.stdout.close()
