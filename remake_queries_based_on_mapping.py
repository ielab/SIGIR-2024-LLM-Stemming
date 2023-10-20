import json
import openai
import torch
import argparse
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader
import os
from tqdm import tqdm
from pyserini.analysis import Analyzer, get_lucene_analyzer
import re
def process_entities(entities):
    processed_entities = []
    #make lower case
    #roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
    #                  'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx',
    #                  'xxi', 'xxii']
    #entities = [entity for entity in entities if
                         #not any(rn in entity.split('.')[0].lower() for rn in roman_numerals)]

    entities = [entity.lower() for entity in entities]
    for entity in entities:
        # Use regex to find and remove any numbering pattern (e.g., "1. ", "2. ", etc.)
        # and an optional asterisk '*' at the beginning.

        cleaned_entity = re.sub(r'^(\d+\.\s*|([ivxlc]+\.\s*)|[•\*]\s*)', '', entity)

        # Check if there are parentheses and extract the content before it.
        match = re.search(r'^(.*?)\s*\((.*?)\)', cleaned_entity)
        if match:
            # Add the content outside the parentheses.
            processed_entities.append(match.group(1).strip())
            # Add the full entity.
            processed_entities.append(cleaned_entity.strip())
        else:
            # Check if there is a colon and extract the content before it.
            match = re.search(r'^(.*?)\s*:', cleaned_entity)
            if match:
                # Add both parts (the one before and the full entity with the explanation) to the entities.
                processed_entities.append(match.group(1).strip())
            processed_entities.append(cleaned_entity.strip())
    processed_entities = [entity for entity in processed_entities if len(entity)>1]
    return processed_entities


def process_entities_exp(entities):
    processed_entities = {}
    current_key = None  # To keep track of the last key we found
    entities = [entity.lower() for entity in entities]

    for entity in entities:
        # Skip empty entries
        if not entity.strip():
            continue

        # Use regex to find and remove any numbering pattern (e.g., "1. ", "2. ", etc.)
        cleaned_entity = re.sub(r'^\d+\.\s*', '', entity)
        cleaned_entity = re.sub(r'^\*', '', cleaned_entity).strip()  # Remove asterisks at the beginning

        # Check if we found a new key
        if ':' in cleaned_entity:
            parts = cleaned_entity.split(':')
            current_key = parts[0].strip()
            if len(parts) > 1 and parts[1].strip():
                # Process values in the same element
                processed_entities[current_key] = [x.strip() for x in parts[1].split(',')]
            else:
                processed_entities[current_key] = []
            continue

        # If there is no current key, we can't do anything with this entry
        if current_key is None:
            continue

        # By this point, cleaned_entity should be a value related to current_key
        processed_entities[current_key].append(cleaned_entity)
    if "herein" in processed_entities:
        del processed_entities["herein"]

    return processed_entities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="trec_covid")
    parser.add_argument("--model", type=str, default="openai")
    parser.add_argument("--type", type=str, default="vocab")
    parser.add_argument("--vocab_model", type=str, default="porter")
    parser.add_argument("--type_2", type=str, default="replace", help="replace or append")
    arg = parser.parse_args()
    type = arg.type
    type_2 = arg.type_2
    query_file = os.path.join("topics", arg.dataset, "default.tsv")
    vocab_model = arg.vocab_model
    if arg.model=="porter":
        analyzer = Analyzer(get_lucene_analyzer(stemmer="porter", stopwords=False))
        lines = []
        with open(query_file, "r") as f:
            for line in f:
                try:
                    qid, query = line.strip().split("\t")
                except:
                    qid, query = line.strip().split(" ", 1)
                    print(query)
                query_tokens = analyzer.analyze(query)
                lines.append(qid + "\t" + " ".join(query_tokens) + "\n")
        output_file = os.path.join("topics", arg.dataset, arg.model + ".tsv")
        with open(output_file, "w") as f:
            f.writelines(lines)
        return None
    if arg.model == "krovetz":
        analyzer = Analyzer(get_lucene_analyzer(stemmer="krovetz", stopwords=False))
        lines = []
        with open(query_file, "r") as f:
            for line in f:
                try:
                    qid, query = line.strip().split("\t")
                except:
                    qid, query = line.strip().split(" ", 1)
                    print(query)
                query_tokens = analyzer.analyze(query)
                lines.append(qid + "\t" + " ".join(query_tokens) + "\n")
        output_file = os.path.join("topics", arg.dataset, arg.model + ".tsv")
        with open(output_file, "w") as f:
            f.writelines(lines)
        return None

    if type == "vocab":
        analyzer = Analyzer(get_lucene_analyzer(stemming=False, stopwords=False))
        mapping = os.path.join("output", f"stemmed_{type}_" + arg.model + "_" + arg.dataset + ".jsonl")
        with open(mapping, "r") as f:
            mapping_dict = json.load(f)
        lines = []
        with open(query_file, "r") as f:
            for line in f:
                try:
                    qid, query = line.strip().split("\t")
                except:
                    qid, query = line.strip().split(" ", 1)
                    print(query)
                query_tokens = analyzer.analyze(query)
                new_query_tokens = []
                for token in query_tokens:
                    if token in mapping_dict:
                        new_query_tokens.extend(mapping_dict[token])
                    #else:
                        #print(qid, token)
                        #new_query_tokens.append(token)
                lines.append(qid + "\t" + " ".join(new_query_tokens) + "\n")

        output_file = os.path.join("topics", arg.dataset, arg.model + f"_{type}.tsv")
        with open(output_file, "w") as f:
            f.writelines(lines)
    else:
        analyzer = Analyzer(get_lucene_analyzer(stemmer="porter", stopwords=False))
        mapping_file = os.path.join("topics",  arg.dataset + "/" + f"{arg.model}_{type}" + ".jsonl")
        #analyser2 = Analyzer(get_lucene_analyzer(stemming=False, stopwords=False))
        mapping_dict = {}
        if type == "context":
            with open(os.path.join(mapping_file), "r") as f:
                for line in f:
                    data = json.loads(line)
                    new_contents = data["contents"]
                    mapping_dict[data["id"]] = new_contents
            lines = []
            with open(query_file, "r") as f:
                for line in f:
                    try:
                        qid, query = line.strip().split("\t")
                    except:
                        qid, query = line.strip().split(" ", 1)
                        print(query)
                    query = query.lower()
                    new_query = query
                    if qid in mapping_dict:
                        new_query = mapping_dict[qid]
                    lines.append(qid + "\t" + new_query + "\n")
                    output_file = os.path.join("topics", arg.dataset, arg.model + "_context.tsv")
                    with open(output_file, "w") as f:
                        f.writelines(lines)

        if type == "entities":
            if vocab_model!="porter":
                analyzer = Analyzer(get_lucene_analyzer(stemming=False, stopwords=False))
                vocab_file = os.path.join("output", f"stemmed_vocab_" + vocab_model + "_" + arg.dataset + ".jsonl")
                with open(vocab_file, "r") as f:
                    vocab_dict = json.load(f)

            with open(os.path.join(mapping_file), "r") as f:
                for line in f:
                    data = json.loads(line)
                    entities = data["entities"]
                    processed_entities = process_entities(entities)
                    mapping_dict[data["id"]] = processed_entities
            lines = []
            with open(query_file, "r") as f:
                for line in f:
                    try:
                        qid, query = line.strip().split("\t")
                    except:
                        qid, query = line.strip().split(" ", 1)
                        print(query)
                    query = query.lower()
                    query_tokens = []

                    token_entities = mapping_dict[qid]
                    token_entities = sorted(token_entities, key=len, reverse=True)


                    for entity in token_entities:
                        count = query.count(entity)
                        if count>0:
                            #query_tokens.extend(analyzer.analyze(entity))
                            query_tokens.extend([entity] * count)
                            if arg.type_2=="replace":
                                query = query.replace(entity, "")
                            #query = query.replace(entity, "")
                    if vocab_model=="porter":
                        query_tokens.extend(analyzer.analyze(query))

                    else:
                        tem_tokens = analyzer.analyze(query)
                        for token in tem_tokens:
                            if token in vocab_dict:
                                query_tokens.extend(vocab_dict[token])
                            else:
                                query_tokens.append(token)

                    lines.append(qid + "\t" + " ".join(query_tokens) + "\n")
                    output_file = os.path.join("topics", arg.dataset, arg.model + "_" + vocab_model + "_" + type_2 + "_entities.tsv")
                    with open(output_file, "w") as f:
                        f.writelines(lines)


if __name__ == "__main__":
    main()

