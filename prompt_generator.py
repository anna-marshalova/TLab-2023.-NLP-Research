import random
from example import Example


class PromptGenerator:
    def __init__(self, examples_raw, prefix_indices=None, prefixes=None, num_problems=1, start=0, end=20):
        self.examples = [Example(ex) for ex in examples_raw]
        self.num_problems = num_problems
        self.start = start
        self.end = end
        self.prefixes, self.prefix_indices = self.chose_prefixes(prefixes, prefix_indices)
        self.prefix = "\n".join(self.prefixes)

    def chose_prefixes(self, prefixes, prefix_indices):
        if not prefixes:
            if not prefix_indices:
                prefix_indices = self.chose_random_prefixes()
            prefixes = [str(self.examples.pop(prefix_index)) for prefix_index in prefix_indices]
        return prefixes, prefix_indices

    def chose_random_prefixes(self):
        indices = list(range(0, self.start)) + list(range(self.end, len(examples)))
        return random.sample(indices, self.num_problems)

    def generate_prompt(self, question):
        return f'{self.prefix}\nQ: {question} \nA:'

    def generate_prompts(self):
        examples = self.examples[self.start:self.end]
        questions = [ex.question for ex in examples]
        answers = [ex.answer for ex in examples]
        prompts = [self.generate_prompt(q) for q in questions]
        assert len(questions) == len(answers) == len(prompts)
        print(f'{len(answers)} prompt(s) generated.')
        return prompts, answers
