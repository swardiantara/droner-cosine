from models.TENER import TENER
from fastNLP.embeddings import CNNCharEmbedding, BertEmbedding
from fastNLP import cache_results
from fastNLP import Trainer, GradientClipCallback, WarmupCallback
from torch import optim
from fastNLP import SpanFPreRecMetric, BucketSampler
from fastNLP.io.pipe.conll import OntoNotesNERPipe, Conll2003NERPipe
from fastNLP.embeddings import StaticEmbedding, StackEmbedding, LSTMCharEmbedding, ElmoEmbedding
from modules.TransformerEmbedding import TransformerCharEmbed
# from modules.pipe import Conll2003NERPipe, DroneNERPipe
import os

import argparse
from modules.callbacks import EvaluateCallback

device = 0
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='conll2003', choices=['conll2003', 'en-ontonotes', 'drone'])
parser.add_argument('--char_embed', type=str, choices=['lstm', 'cnn', 'adatrans'], default='lstm', help='Char Embedding methods')
parser.add_argument('--word_embed', type=str, choices=['glove', 'bert', 'elmo'], default='static', help='Type of Word Embdding used')
parser.add_argument('--output_dir', type=str, help="Where to store the output files")
parser.add_argument('--overwrite_output_dir', action="store_true", help="Overwrite the content of the output directory")
parser.add_argument('--encoder', type=str, choices=['transformer', 'lstm'], help="Encoder Architecture used to perform computation")
parser.add_argument('--attention_type', type=str, default='transformer', choices=['cosine', 'transformer', 'adatrans'], help="Attention Type used to compute attention score")
parser.add_argument('--scaled_attention', action='store_true', help="Wether to scale the attention score using sqrt(d_k)")
parser.add_argument('--decoder', type=str, choices=['crf', 'softmax'], help="Type of decoder used to interpret the logits")

args = parser.parse_args()
output_dir = os.path.join(args.output_dir, '{}_{}_{}_{}'.format(args.attention_type, args.char_embed, args.word_embed, args.decoder))
dataset = args.dataset

if dataset == 'conll2003':
    n_heads = 14
    head_dims = 128
    num_layers = 2
    lr = 0.0009
    # char_type = 'lstm'
    # word_type = 'glove'
elif dataset == 'en-ontonotes':
    n_heads =  8
    head_dims = 96
    num_layers = 2
    lr = 0.0007
    # attn_type = 'adatrans'
    # char_type = 'adatrans'
elif dataset == 'drone':
    n_heads = 6
    head_dims = 128
    num_layers = 2
    lr = 0.0009
    # attn_type = 'cosine'
    # char_type = 'lstm'
    # word_type = 'glove'

# Config     
pos_embed = None
attn_type = args.attention_type
word_type = args.word_embed
char_type = args.char_embed
scale = True if args.scaled_attention else False
#########hyper
batch_size = 8
warmup_steps = 0.01
after_norm = 1
model_type = 'transformer'
normalize_embed = True
#########hyper

dropout=0.15
fc_dropout=0.4

encoding_type = 'bio'
name = 'caches/{}_{}_{}_{}_{}.pkl'.format(dataset, model_type, encoding_type, char_type, normalize_embed)
d_model = n_heads * head_dims
dim_feedforward = int(2 * d_model)



@cache_results(name, _refresh=False)
def load_data():
    # ????????????
    if dataset == 'conll2003':
        # conll2003???lr????????????0.002
        paths = {'test': "./data/conll2003/test.txt",
                 'train': "./data/conll2003/train.txt",
                 'dev': "./data/conll2003/dev.txt"}
        data = Conll2003NERPipe(encoding_type=encoding_type).process_from_file(paths)
    elif dataset == 'en-ontonotes':
        # ??????????????????????????????train.txt, test.txt, dev.txt?????????
        paths = '../data/en-ontonotes/english'
        data = OntoNotesNERPipe(encoding_type=encoding_type).process_from_file(paths)
    elif dataset == 'drone':
        paths = {
            'train': "./data/drone/train.txt",
            'dev': "./data/drone/test.txt",
            'test': "./data/drone/test.txt",
        }
        data = Conll2003NERPipe(encoding_type=encoding_type).process_from_file(paths)
    char_embed = None
    data.rename_field('words', 'chars')
    if char_type == 'cnn':
        char_embed = CNNCharEmbedding(vocab=data.get_vocab('words'), embed_size=30, char_emb_size=30, filter_nums=[30],
                                      kernel_sizes=[3], word_dropout=0, dropout=0.3, pool_method='max'
                                      , include_word_start_end=False, min_char_freq=2)
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
        word_embed = ElmoEmbedding(vocab=data.get_vocab('words'), model_dir_or_name='en-original', layers='mix', requires_grad=False,
                 word_dropout=0.0, dropout=0.5, cache_word_reprs=False)
        word_embed.set_mix_weights_requires_grad()
    elif word_type == 'glove':
        word_embed = StaticEmbedding(vocab=data.get_vocab('words'),
                                    model_dir_or_name='en-glove-6b-100d',
                                    requires_grad=True, lower=False, word_dropout=0, dropout=0.5,
                                    only_norm_found_vector=normalize_embed)
    if char_embed is not None:
        embed = StackEmbedding([word_embed, char_embed], dropout=0, word_dropout=0.02)
    else:
        word_embed.word_drop = 0.02
        embed = word_embed

    
    return data, embed

data_bundle, embed = load_data()
print(data_bundle)

model = TENER(tag_vocab=data_bundle.get_vocab('target'), embed=embed, num_layers=num_layers,
                       d_model=d_model, n_head=n_heads,
                       feedforward_dim=dim_feedforward, dropout=dropout,
                        after_norm=after_norm, attn_type=attn_type,
                       bi_embed=None,
                        fc_dropout=fc_dropout,
                       pos_embed=pos_embed,
              scale=scale)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

callbacks = []
clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
evaluate_callback = EvaluateCallback(data_bundle.get_dataset('test'))

if warmup_steps>0:
    warmup_callback = WarmupCallback(warmup_steps, schedule='linear')
    callbacks.append(warmup_callback)
callbacks.extend([clip_callback, evaluate_callback])

trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer, batch_size=batch_size, sampler=BucketSampler(),
                  num_workers=2, n_epochs=15, dev_data=data_bundle.get_dataset('dev'),
                  metrics=SpanFPreRecMetric(tag_vocab=data_bundle.get_vocab('target'), encoding_type=encoding_type, f_type = 'micro', only_gross=False),
                  dev_batch_size=batch_size*5, callbacks=callbacks, device=device, test_use_tqdm=False,
                  use_tqdm=True, print_every=300, save_path=None)
trainer.train(load_best_model=False)
