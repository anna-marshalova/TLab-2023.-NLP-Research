import torch
from tqdm.notebook import tqdm


class Generator:
    def __init__(self, model, tokenizer, device='cuda', add_phrase=None):
        self.device = device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.stop_token_ids = tokenizer(tokenizer.eos_token)['input_ids']
        if add_phrase:
            # Векторизуем фразу,чтобы потом добавлять ее к сгенерированному тексту
            self.answer_inputs = tokenizer(add_phrase, return_tensors='pt').to(self.device)
        else:
            self.answer_inputs = None

    def generate(self, prompt, answer, num_iter, do_sample, temperature):
        solutions = []
        inputs = self.tokenizer(prompt, padding=True, return_tensors='pt').to(self.device)
        for i in tqdm(range(num_iter), leave=False):
            output = self.model.generate(inputs['input_ids'], do_sample=do_sample, temperature=temperature,
                                         max_new_tokens=50,
                                         repetition_penalty=10, stop_token_ids=self.stop_token_ids)
            if self.answer_inputs:
                # Добавляем фразу к тексту и генерируем еще 1 токен
                output = torch.cat((output, self.answer_inputs['input_ids']), dim=1)
                output = self.model.generate(output, do_sample=True, max_new_tokens=1)
            solution = self.tokenizer.decode(output[0])
            solutions.append(solution)
            print(f'{solution}\nCorrect answer: {answer}\n')
        return solutions

    def generate_batch(self, prompts, answers, num_iter=1):
        solutions = []
        for prompt, answer in zip(tqdm(prompts), answers):
            solutions.append(self.generate(prompt, answer, num_iter, do_sample=True, temperature=1))
        return solutions
