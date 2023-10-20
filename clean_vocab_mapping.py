import json
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="trec_covid")
parser.add_argument("--model", type=str, default="openai")

arg = parser.parse_args()


dataset = arg.dataset
model = arg.model
def is_number_or_number_list(term):
    punctuations = [",", ".", ";", ":", "!", "'"]

    # Split the term by punctuations
    segments = [term]
    for punc in punctuations:
        temp_segments = []
        for seg in segments:
            temp_segments.extend(seg.split(punc))
        segments = temp_segments

    # Check if all segments can be converted to a float
    for seg in segments:
        try:
            float(seg)
        except ValueError:
            return False
    return True

def check_correct(original_term, stemmed_terms):
    #so if original term do not have punctuations, then stemmed terms should not have punctuations
    punctuations = ",.;:!'"
    results = []

    for stemmed_term in stemmed_terms:
        valid = True
        for punc in punctuations:
            if punc not in original_term and punc in stemmed_term:
                valid = False
                break
        if valid:
            results.append(stemmed_term)
    return results



mapping_dict = {}
with open(f"output/stemmed_vocab_{model}.jsonl") as f:
    for line in f:
        current_dict = json.loads(line)
        for term in current_dict:
            #check correct
            stemmed_terms = current_dict[term]
            checked_terms = check_correct(term, stemmed_terms)
            fina_checked = []
            for stemmed_term in checked_terms:
                if (stemmed_term == "program") and ("program" not in term):
                    continue
                fina_checked.append(stemmed_term)

            if len(checked_terms) > 0:
                mapping_dict[term] = fina_checked

print(len(mapping_dict))

dataset=f"indexes/{dataset}"
index_reader = IndexReader(dataset)#.from_prebuilt_index(dataset)

print(index_reader.stats())
    # get all the terms from the index
terms = index_reader.terms()

output_dict = {}
term_set = set()
count = 0
counter1 = 0
for term in tqdm(terms):
    current_term = term.term
    if is_number_or_number_list(current_term):
        continue
    counter1 +=1
    if current_term in mapping_dict:
        output_dict[current_term] = mapping_dict[current_term]
        count += 1
print(f"there should be {counter1} terms in output")
print(f"there are {count} terms in the output dict")
print(f"there are {counter1-count} terms not in the output dict")


out_file = f"output/stemmed_vocab_{model}_{dataset.split('/')[-1]}.jsonl"

with open(out_file, "w") as fw:
    json.dump(output_dict, fw)
