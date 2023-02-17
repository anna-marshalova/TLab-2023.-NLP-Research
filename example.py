import re
import dataset, sample


class Example:
    def __init__(self, example):
        self.__METATEXT = re.compile('#### \d+<\|endoftext\|>')
        self.__EQUATION = re.compile('<<.*?>>')
        self.question = example['question'].rstrip()
        self.thought = self.process_thought(example['answer'])
        self.answer = example.get('num_answer', dataset.extract_answer(example['answer']))

    def process_thought(self, thought):
        thought = re.sub('\n', ' ', thought)
        thought = re.sub('\n', ' ', thought)
        thought = self.__EQUATION.sub('', thought)
        return self.__METATEXT.sub('', thought)

    def __str__(self):
        return f'Q: {self.question}\nA: {self.thought}\nThe answer is {self.answer}.{tokenizer.eos_token}\n\n'
        # return f'Q: {self.question} \nA: Let’s think step by step. {self.thought} \nThe answer is {self.answer}.{tokenizer.eos_token}'
