import random
import torch
from tqdm.notebook import tqdm, trange


class Generator:
    def __init__(self, model, tokenizer, examples=None, prefix_indices=None, prefixes=None, add_phrase=None,
                 device='cuda'):
        assert examples or prefix_indices
        self.examples = examples
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.stop_token_ids = tokenizer(tokenizer.eos_token)['input_ids']
        self.prefixes = prefixes
        self.prefix_indices = prefix_indices
        if add_phrase:
            # Векторизуем фразу,чтобы потом добавлять ее к сгенерированному тексту
            self.answer_inputs = tokenizer(add_phrase, return_tensors='pt').to(self.device)
        else:
            self.answer_inputs = None

    def chose_prefixes(self, num_problems):
        if not self.prefixes:
            if not self.prefix_indices:
                prefix_indices = random.sample(range(len(self.examples)), num_problems)
            else:
                prefixes = self.prefixes
            prefixes = [str(self.examples.pop(prefix_index)) for prefix_index in prefix_indices]
        return prefixes

    def generate_prompt(self, question, prefix):
        return f'{prefix}\nQ: {question} \nA:'

    def generate(self, prompt, answer, num_iter, do_sample, **kwargs):
        solutions = []
        inputs = self.tokenizer(prompt, padding=True, return_tensors='pt').to(self.device)
        if num_iter > 1:
            iterator = trange(num_iter, leave=False)
        else:
            iterator = range(1)
        for i in iterator:
            output = self.model.generate(inputs['input_ids'],
                                         max_new_tokens=50,
                                         repetition_penalty=10, stop_token_ids=self.stop_token_ids,
                                         do_sample=do_sample, **kwargs)
            if self.answer_inputs:
                # Добавляем фразу к тексту и генерируем еще 1 токен
                output = torch.cat((output, self.answer_inputs['input_ids']), dim=1)
                output = self.model.generate(output, do_sample=True, max_new_tokens=1)
            solution = self.tokenizer.decode(output[0])
            solutions.append(solution)
            print(f'{solution}\nCorrect answer: {answer}\n')
        return solutions

    def generate_batch(self, questions, answers, ensemble_size=1, ensembling='sampling', num_problems=1,
                       do_sample=False,  **kwargs):
        solutions = []
        if ensemble_size > 1 and ensembling == 'prompt':
            prefixes = self.chose_prefixes(ensemble_size)
            for question, answer in zip(tqdm(questions), answers):
                solutions.append([])
                for prefix in tqdm(prefixes, leave=False):
                    prompt = self.generate_prompt(question, prefix)
                    solutions[-1].extend(self.generate(prompt, answer, 1, do_sample=do_sample, **kwargs))
        else:
            prefixes = self.chose_prefixes(num_problems)
            prefix = "\n".join(prefixes)
            for question, answer in zip(tqdm(questions), answers):
                prompt = self.generate_prompt(question, prefix)
                solutions.append(self.generate(prompt, answer, ensemble_size, do_sample=do_sample, **kwargs))
        return solutions
