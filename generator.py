import torch
from tqdm.notebook import tqdm


def generate(prompt, answer, model, tokenizer, num_iter=1, do_sample=True, add_phrase=None, temperature=1,
             device='cuda'):
    solutions = []
    inputs = tokenizer(prompt, padding=True, return_tensors='pt').to(device)
    stop_token_ids = tokenizer(tokenizer.eos_token)['input_ids']
    if add_phrase:
        # Векторизуем фразу,чтобы потом добавлять ее к сгенерированному тексту
        answer_inputs = tokenizer(add_phrase, return_tensors='pt').to(device)
    for i in tqdm(range(num_iter), leave=False):
        output = model.generate(inputs['input_ids'], do_sample=do_sample, temperature=temperature, max_new_tokens=50,
                                repetition_penalty=10, stop_token_ids=stop_token_ids)
        if add_phrase:
            # Добавляем фразу к тексту и генерируем еще 1 токен
            output = torch.cat((output, answer_inputs['input_ids']), dim=1)
            output = model.generate(output, do_sample=True, max_new_tokens=1)
        solution = tokenizer.decode(output[0])
        solutions.append(solution)
        print(f'{solution}\nCorrect answer: {answer}\n')
    return solutions
