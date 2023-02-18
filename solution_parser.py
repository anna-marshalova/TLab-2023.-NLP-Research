import re
from collections import Counter
from utils import sort_thoughts_by_length
from example import Example


class SolutionParser:
    def __init__(self, consistency=None):
        self.EOS = '</s>'
        self.NUM_PATTERN = '\d*[.,]?\d+'
        self.NUM_RE = re.compile(self.NUM_PATTERN)
        self.ANSWER_RE = re.compile(f'answer is \$?({self.NUM_PATTERN})')
        self.PARSE_RE = re.compile('Q:(.*?)A:(.*)')
        if consistency == 'self':
            self.consistency = self.self_consistency
        elif consistency == 'complexity':
            self.consistency = self.complexity_based_consistency
        else:
            self.consistency = None

    def extract_answer(self, thought):
        possible_answers = self.ANSWER_RE.findall(thought)
        if possible_answers:
            return possible_answers[-1]
        possible_answers = self.NUM_RE.findall(thought)
        if possible_answers:
            return possible_answers[-1]
        return None

    def parse_solution(self, output, num_problems):
        solution = output.split(self.EOS)[num_problems]
        solution = solution.split('Q:')[1].split('A:')
        question = solution[0]
        thought = solution[-1]
        answer = self.extract_answer(thought)
        return {'question': question, 'answer': thought, 'num_answer': answer}

    def parse_solutions(self, solutions, num_problems=1):
        if self.consistency:
            return [[Example(self.parse_solution(sample, num_problems)) for sample in samples] for samples in solutions]
        else:
            return [Example(self.parse_solution(sample, num_problems)) for samples in solutions for sample in samples]

    def choose_answers(self, solutions):
        parsed_solutions = self.parse_solutions(solutions)
        chosen_answers = []
        for sampled_solutions in parsed_solutions:
            sampled_solutions = [s for s in sampled_solutions if s.answer]
            chosen_answers.append(self.consistency(sampled_solutions))
        return chosen_answers

    def self_consistency(self, sampled_solutions):
        sampled_answers = Counter([s.answer for s in sampled_solutions]).most_common()
        if sampled_answers[0][1] == 1:
            print('There are no most frequent answers. Using complexity based consistency.')
            return self.complexity_based_consistency(sampled_solutions)
        for s in sampled_solutions:
            if s.answer == sampled_answers[0][0]:
                return s

    def complexity_based_consistency(self, sampled_solutions):
        sampled_answers_lengths = sort_thoughts_by_length(sampled_solutions)
        best_solution = sampled_solutions[sampled_answers_lengths[0][0]]
        return best_solution

    def compare_answers(self, parsed_solutions, answers):
        for pred_solution, answer in zip(parsed_solutions, answers):
            print(pred_solution.answer, answer)
        if pred_solution.answer == answer:
            print(pred_solution)

    def print_solutions(self, solutions):
        for idx, solution in enumerate(solutions):
            print(f'{idx+1}:{solution}')