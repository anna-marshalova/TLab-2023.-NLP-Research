import re
from collections import Counter
from utils import sort_thoughts_by_complexity
from example import Example


class SolutionParser:
    def __init__(self, consistency=None):
        """
        :param consistency: str Метод ансамблирования: self-consistnecy или complexity based consistency. По умолчанию отсутствует.
        """
        self.EOS = '</s>'
        # Число
        self.NUM_PATTERN = '\d*[.,]?\d+'
        self.NUM_RE = re.compile(self.NUM_PATTERN)
        # Число после фразы answer is
        self.ANSWER_RE = re.compile(f'answer is \$?({self.NUM_PATTERN})')
        self.PARSE_RE = re.compile('Q:(.*?)A:(.*)')
        if consistency == 'self':
            self.consistency = self.self_consistency
        elif consistency == 'complexity':
            self.consistency = self.complexity_based_consistency
        else:
            self.consistency = None

    def extract_answer(self, thought):
        """Извлечение ответа из текста решения
        :param thought: str Решение
        :return: str Ответ
        """
        # Ищем числа после фразы answer is
        possible_answers = self.ANSWER_RE.findall(thought)
        if possible_answers:
            return possible_answers[-1]
        # Ищем последнее число в тексте
        possible_answers = self.NUM_RE.findall(thought)
        if possible_answers:
            return possible_answers[-1]
        return None

    def parse_solution(self, output, num_problems):
        """Извлечение решения из сгенерированного текста
        :param output: str Сгенерированный текст вместе с промптом
        :param num_problems: int Количество примеров в промпте
        :return: Dict[str] Вопрос, решение и ответ
        """
        # Искомая задача следует за примера из промпта
        solution = output.split(self.EOS)[num_problems]
        # Символы начала вопроса или ответа (Q и A) могут быть сгенерированы несколько раз.
        # Возьмем первый вопрос (который был задан в промпте) и последний ответ на него
        solution = solution.split('Q:')[1].split('A:')
        question = solution[0]
        thought = solution[-1]
        answer = self.extract_answer(thought)
        return {'question': question, 'answer': thought, 'num_answer': answer}

    def parse_solutions(self, solutions, num_problems=1):
        """Преобразование сгенерированного текста в удобный формат
        :param solutions: List[List[str]] Сгенерированные тексты
        :param num_problems: int Количество примеров в промпте
        :return: List Сгенерированные тексты как эксземпляры класса Example
        """
        if self.consistency:
            return [[Example(self.parse_solution(sample, num_problems)) for sample in samples] for samples in solutions]
        else:
            return [Example(self.parse_solution(sample, num_problems)) for samples in solutions for sample in samples]

    def choose_answers(self, solutions):
        """Выбор одного из ответов
        :param solutions: List[List[str]] Сгенерированные тексты
        :return: List[Tuple[str, str] Список вопросов и выбранных для них ответов
        """
        parsed_solutions = self.parse_solutions(solutions)
        chosen_answers = []
        for sampled_solutions in parsed_solutions:
            sampled_solutions = [s for s in sampled_solutions if s.answer]
            chosen_answers.append(self.consistency(sampled_solutions))
        return chosen_answers

    def self_consistency(self, sampled_solutions):
        """Выбор самого частого ответа
        :param sampled_solutions: List[str] Решения для одной задчаи
        :return: str Выбранный ответ
        """
        sampled_answers = Counter([s.answer for s in sampled_solutions]).most_common()
        if sampled_answers[0][1] == 1:
            print('There are no most frequent answers. Using complexity based consistency.')
            return self.complexity_based_consistency(sampled_solutions)
        for s in sampled_solutions:
            if s.answer == sampled_answers[0][0]:
                return s

    def complexity_based_consistency(self, sampled_solutions):
        """Выбор самого сложного решения
        :param sampled_solutions: List[str] Решения для одной задчаи
        :return: str Выбранный ответ
        """
        sampled_answers_lengths = sort_thoughts_by_complexity(sampled_solutions)
        best_solution = sampled_solutions[sampled_answers_lengths[0][0]]
        return best_solution

    def compare_answers(self, parsed_solutions, answers):
        """Вывод пар предсказанный ответ - правильный ответ. Если ответы совпадают, выводится текст решения
        :param parsed_solutions: List[Example] Сгенеированные решения
        :param answers: List[str] Правильные ответы
        """
        for pred_solution, answer in zip(parsed_solutions, answers):
            print(pred_solution.answer, answer)
        if pred_solution.answer == answer:
            print(pred_solution)

    def print_solutions(self, solutions):
        """Вывод решений на экран
        :param solutions: List Сгенеированные решения
        :return:
        """
        for idx, solution in enumerate(solutions):
            print(f'{idx + 1}:{str(solution)}')
