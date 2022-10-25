import pandas as pd

def to_conll(dataframe, split_type):
    filename = "train.txt" if split_type == "train" else "test.txt"
    prev_sentence_id = 1
    # var = true_val if condition else false_val
    with open(filename, "w") as f_out:
        for _, line in dataframe.iterrows():
            current_sentence_id = line['sentence_id']
            word = line["words"]
            tag = line["labels"]
            if(current_sentence_id == prev_sentence_id):
                # for sentence_id, word, tag in zip(line['sentence_id'], line["words"], line["labels"]):
                print("{} X X {}".format(word, tag), file=f_out)
            else: # Newline for sentence separator
                print("\n{} X X {}".format(word, tag), file=f_out)
                prev_sentence_id = current_sentence_id
            # print(file=f_out)

def main():
    train_path = './droner_train_2.csv'
    test_path = './droner_test_2.csv'

    train_df = pd.read_csv(train_path, encoding= 'unicode_escape')
    test_df = pd.read_csv(test_path, encoding= 'unicode_escape')
    # train_df = train_df.append(test_df)

    to_conll(train_df, "train")
    to_conll(test_df, "test")
# Sesuaikan dengan format CoNLL2003 data.
# Buat Pipeline sesuai data (karena hanya ada 2 kolom ['word', 'tag'])
# Bikin pipeline sendiri, atau ikuti kode FastNLP
# 
# FastNLP ada BertEmbedding. Bisa dipakai di layer Embedding BERT-Cosine-CRF
# https://fastnlp.readthedocs.io/zh/latest/fastNLP.embeddings.bert_embedding.html

if __name__ == "__main__":
    main()