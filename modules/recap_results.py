# Scritp to recap experimental results
import os
import json
import pandas as pd


class EvaluationDict(dict):
    def __init__(self):
        self = dict()

    def add_attribute(self, key, value):
        self[key] = value


def txt_to_json():
    datasets = ['drone-polysemi']
    attentions = ['cosine', 'transformer', 'adatrans']
    working_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    print(working_dir)

    for dataset in datasets:
        # output_dir = os.path.join('output', dataset)
        for attention in attentions:
            output_dir = os.path.join('output-2', dataset, attention)
            # print(output_dir)
            files = os.listdir(output_dir)
            for filename in files:
                file_ext = filename.split('.')[-1]
                if not file_ext == 'txt':
                    continue
                file_path = os.path.join(output_dir, filename)
                filename_json = filename.split('.')[0] + '.json'
                file = open(file_path)
                param = EvaluationDict()
                scaled = filename.split('-')[0]
                param.add_attribute('scaled', scaled)
                per_entity = list(dict())
                micro_average = dict()
                for i, line in enumerate(file):
                    # construct the .json file
                    # Micro-average score
                    if (i == 7):
                        scheme = line.split(':')[1].strip()
                        param.add_attribute('scheme', scheme)
                    elif (i == 8):
                        pos_embed = line.split(':')[1].strip()
                        param.add_attribute('pos_embed', pos_embed)
                    elif (i == 9):
                        char_embed = line.split(':')[1].strip()
                        param.add_attribute('char_embed', char_embed)
                    elif (i == 10):
                        word_embed = line.split(':')[1].strip()
                        param.add_attribute('word_embed', word_embed)
                    elif (i == 11):
                        attention = line.split(':')[1].strip()
                        param.add_attribute('attention', attention)
                    elif i == 371:
                        scores = line.split(':')[1].strip().split(',')
                        f1 = scores[0].strip().split('=')[1]
                        precision = scores[1].split('=')[1]
                        recall = scores[2].split('=')[1]
                        micro_average = {
                            'f': f1,
                            'pre': precision,
                            'rec': recall,
                        }
                    elif i == 374:
                        scores = line.split(':')[1].strip().split(',')
                        scores = [score.strip() for score in scores]
                        issue = EvaluationDict()
                        component = EvaluationDict()
                        action = EvaluationDict()
                        state = EvaluationDict()
                        parameter = EvaluationDict()
                        function = EvaluationDict()
                        for score in scores:
                            entity_name = score.split('=')[0].split('-')
                            if (len(entity_name) > 1):
                                metric_name = score.split('-')[0]
                                value = score.split('=')[1]
                                if entity_name[1] == 'issue':
                                    issue.add_attribute(metric_name, value)
                                elif entity_name[1] == 'component':
                                    component.add_attribute(metric_name, value)
                                elif entity_name[1] == 'action':
                                    action.add_attribute(metric_name, value)
                                elif entity_name[1] == 'state':
                                    state.add_attribute(metric_name, value)
                                elif entity_name[1] == 'parameter':
                                    parameter.add_attribute(metric_name, value)
                                elif entity_name[1] == 'function':
                                    function.add_attribute(metric_name, value)
                        # Per-entity Evaluation
                        per_entity = [
                            {'entity': 'issue', 'score': issue},
                            {'entity': 'state', 'score': state},
                            {'entity': 'action', 'score': action},
                            {'entity': 'function', 'score': function},
                            {'entity': 'parameter', 'score': parameter},
                            {'entity': 'component', 'score': component},
                        ]
                results = {
                    "parameter": param,
                    "per_entity": per_entity,
                    "micro_avg": micro_average
                }

                with open(os.path.join(output_dir, filename_json), 'w') as file:
                    json.dump(results, file)
                file.close()


def json_to_csv():
    datasets = ['drone-polysemi']
    attentions = ['cosine', 'transformer', 'adatrans']
    working_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))

    for dataset in datasets:
        # output_dir = os.path.join('output', dataset)
        for attention in attentions:
            evaluation_list = []
            output_dir = os.path.join('output-2', dataset, attention)
            # print(output_dir)
            files = os.listdir(output_dir)
            parent_df = pd.DataFrame()
            for filename in files:
                file_ext = filename.split('.')[1]
                counter = filename.split('.')[0].split('_')[-1]
                file_path = os.path.join(output_dir, filename)
                if file_ext != 'json':
                    continue
                result = open(file_path)
                result = json.load(result)
                row_list = []
                char_embed = result['parameter']['char_embed']
                word_embed = result['parameter']['word_embed']
                scaled = result['parameter']['scaled']
                precision = result['micro_avg']['pre'] if bool(
                    result['micro_avg']) else "-"
                recall = result['micro_avg']['rec'] if bool(
                    result['micro_avg']) else "-"
                f1_score = result['micro_avg']['f'] if bool(
                    result['micro_avg']) else "-"
                row_list.extend([char_embed, word_embed, scaled,
                                attention, counter, f1_score, precision, recall])
                if bool(result["per_entity"]):
                    for entity_type in result['per_entity']:
                        entity_name = entity_type['entity']
                        for metric, value in entity_type['score'].items():
                            row_list.append(value)
                else:
                    empty = ['-'] * 18
                    row_list.extend(empty)
                    # row_list = row_list + empty
                evaluation_list.append(row_list)
            # save to .xlsx
                # child_df = pd.json_normalize(
                #     result, errors='ignore')
                # parent_df = pd.concat([parent_df, child_df])
            dataframe = pd.DataFrame(evaluation_list, index=None, columns=[
                                     "char_embed", "word_embed", "scaled", "attention", "attempt",  "f1-score", "precision", "recall", 'f1-issue', 'prec-issue', 'rec-issue', 'f1-state', 'prec-state', 'rec-state', 'f1-action', 'prec-action', 'rec-action', 'f1-function', 'prec-function', 'rec-function', 'f1-parameter', 'prec-parameter', 'rec-parameter', 'f1-component', 'prec-component', 'rec-component'])
            output_filename = "evaluation_{}.xlsx".format(attention)
            dataframe.to_excel(
                f"{output_dir}/{output_filename}", index=False, encoding='utf-8')


def train_history():
    datasets = ['drone-polysemi']
    attentions = ['cosine']
    working_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))

    for dataset in datasets:
        # output_dir = os.path.join('output', dataset)
        for attention in attentions:
            output_dir = os.path.join('output', dataset, attention)
            # print(output_dir)
            files = os.listdir(output_dir)
            files = [file for file in files if (file.split(
                '.')[0].split('_')[-1] == '1' and file.split('.')[1] == 'txt')]
            print(files)
            train_history_list = []
            for filename in files:
                file_path = os.path.join(output_dir, filename)
                filename_xlsx = filename.split('.')[0] + '.xlsx'
                file = open(file_path)
                num_line = len(file.readlines())
                file.close()
                row_list = []
                scaled = filename.split('-')[0]
                epoch_line = 21
                print('sebelum loop: ', len(row_list))
                print('sebelum loop: ', row_list)
                file = open(file_path)
                for i, line in enumerate(file):
                    # construct the .json file
                    # Micro-average score
                    if i == 15 and num_line < 18:
                        # Failed scenario
                        empty_score = ['-'] * 50
                        row_list.extend(empty_score)
                        continue
                    elif (i == 9):
                        char_embed = line.split(':')[1].strip()
                        row_list.append(char_embed)
                    elif (i == 10):
                        word_embed = line.split(':')[1].strip()
                        row_list.append(word_embed)
                    elif (i == 11):
                        row_list.append(scaled)
                        attention = line.split(':')[1].strip()
                        row_list.append(attention)
                    elif (i == epoch_line and epoch_line < 365):
                        scores = line.split(':')[1].strip().split(',')
                        scores = [score.strip() for score in scores]
                        f1 = scores[0].split('=')[1]
                        row_list.append(f1)
                        epoch_line = epoch_line + 7
                print('sesudah loop: ', len(row_list))
                print('sesudah loop: ', row_list)
                train_history_list.append(row_list)
                file.close()
            # Generate df
            col_names = ["char_embed", "word_embed", "scaled", "attention"]
            for i in range(50):
                col_names.append(str(i+1))
            dataframe = pd.DataFrame(
                train_history_list, index=None, columns=col_names)
            output_filename = "train_history_{}.xlsx".format(attention)
            dataframe.to_excel(
                f"{output_dir}/{output_filename}", index=False, encoding='utf-8')


if __name__ == "__main__":
    # txt_to_json()
    # json_to_csv()
    train_history()
