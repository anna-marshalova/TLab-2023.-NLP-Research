import json
import nltk
nltk.download('punkt')
from math import log

def example_length(example):
  num_sents = len(nltk.sent_tokenize(example.thought))
  num_symbols = len(example.thought)
  return log(num_sents)+log(num_symbols)

def sort_thoughts_by_length(examples):
  thought_lengths = [(idx, example_length(example)) for idx,example in enumerate(examples)]
  return sorted(thought_lengths, key = lambda x:x[1], reverse=True)