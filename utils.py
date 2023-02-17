import os
import json
import nltk
from math import log
nltk.download('punkt')

def example_length(example):
    num_sents = len(nltk.sent_tokenize(example.thought))
    num_symbols = len(example.thought)
    return log(num_sents) + log(num_symbols)


def sort_thoughts_by_length(examples):
    thought_lengths = [(idx, example_length(example)) for idx, example in enumerate(examples)]
    return sorted(thought_lengths, key=lambda x: x[1], reverse=True)

def load_experiment_results(model_name, suffix=''):
    experiment_name = f'{model_name.split("/")[-1]}_{suffix}'
    path = os.path.join('results', f'{experiment_name}.json')
    with open(path) as js_file:
        solutions = json.load(js_file)
    return solutions


def save_experiment(model_name, solutions, suffix=''):
    experiment_name = f'{model_name.split("/")[-1]}_{suffix}'
    path = os.path.join('results', f'{experiment_name}.json')
    with open(path, 'w') as js_file:
        json.dump(solutions, js_file)
