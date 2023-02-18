import re
import dataset, sample


class Example:
    def __init__(self, example, rationale='natural'):
        """
        :param example: Dict[str] Задачи из датасета
        :param rationale: str Формулировка решения: на ествественном языке (natural) или в виде математического выражения (equation)
        """
        # Лишний текст в конце решения
        self.__METATEXT = re.compile('#### \d+<\|endoftext\|>')
        # Регулярное выражение для математических выражений
        self.__EQUATION = re.compile('<<(.*?)>>')
        self.EOS = '</s>'
        self.question = example['question'].rstrip()
        if rationale == 'equation':
            self.thought = self.pick_equations(example['answer'])
        else:
            self.thought = self.process_thought(example['answer'])
        self.answer = example.get('num_answer', dataset.extract_answer(example['answer']))

    def process_thought(self, thought):
        """Удаление лишних сиволов и токенов из текста решения
        :param thought: str Решение
        :return: str Обработанный текст решения
        """
        thought = re.sub('\n', ' ', thought)
        thought = self.__EQUATION.sub('', thought)
        return self.__METATEXT.sub('', thought)

    def pick_equations(self, thought):
        """Составление решения в виде математических выражений
        :param thought: str Решение
        :return: str Решение в виде математических выражений
        """
        equations = self.__EQUATION.findall(thought)
        return '. '.join(equations)

    def __str__(self):
        return f'Q: {self.question}\nA: {self.thought}\nThe answer is {self.answer}.{self.EOS}\n\n'
