import os
import json
import nltk
from math import log

nltk.download('punkt')


def example_complexity(example):
    """Подсчет сложности решения
    :param example:
    :return: float Сложность решения
    """
    num_sents = len(nltk.sent_tokenize(example.thought))
    num_symbols = len(example.thought)
    return log(num_sents) + log(num_symbols)


def sort_thoughts_by_complexity(examples):
    """Сортировка решений по сложности
    :param examples:
    :return: List[Tuple[int, float]]
    """
    thought_lengths = [(idx, example_complexity(example)) for idx, example in enumerate(examples)]
    return sorted(thought_lengths, key=lambda x: x[1], reverse=True)


def load_experiment_results(model_name, suffix=''):
    """Загрузка резульатов из формата json
    :param model_name: str Название модели
    :param suffix: str Название эскперимента
    :return:
    """
    experiment_name = f'{model_name.split("/")[-1]}_{suffix}'
    path = os.path.join('results', f'{experiment_name}.json')
    with open(path) as js_file:
        solutions = json.load(js_file)
    return solutions


def save_experiment_results(solutions, model_name, suffix=''):
    """Сохранение резульатов в формате json
    :param model_name: List[str] Результаты генерации
    :param model_name: str Название модели
    :param suffix: str Название эскперимента
    :return:
    """
    experiment_name = f'{model_name.split("/")[-1]}_{suffix}'
    path = os.path.join('results', f'{experiment_name}.json')
    with open(path, 'w') as js_file:
        json.dump(solutions, js_file)
