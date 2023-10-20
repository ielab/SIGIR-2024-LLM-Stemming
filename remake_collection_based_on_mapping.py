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
    for entity in entities:
        # Use regex to find and remove any numbering pattern (e.g., "1. ", "2. ", etc.)
        cleaned_entity = re.sub(r'^\d+\.\s*', '', entity)
        # Split entity and explanations using ":"
        key, *value = cleaned_entity.split(':')
        if value and value[
            0].strip().lower() != "none":  # Check if there is an explanation available and it is not "none"
            explanations = value[0].split(', ')
            processed_entities[key.strip()] = [explanation.strip() for explanation in explanations]
        else:
            processed_entities[key.strip()] = []
    return processed_entities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="trec_covid")
    parser.add_argument("--model", type=str, default="msmarco_json/")
    parser.add_argument("--type", type=str, default="vocab")
    parser.add_argument("--vocab_model", type=str, default="porter")
    parser.add_argument("--type_2", type=str, default="replace", help="replace or append")
    arg = parser.parse_args()
    vocab_model = arg.vocab_model
    type2 = arg.type_2
    type= arg.type
    collection_folder = arg.dataset
    model = arg.model
    collection_files = [f for f in os.listdir(collection_folder) if f.endswith(".json")]
    if model=="porter":
        analyzer = Analyzer(get_lucene_analyzer(stemmer="porter", stopwords=False))
        collection_folder = f"{collection_folder}"
        output_folder = f"{collection_folder}_porter"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for collection_file in tqdm(collection_files):
            output_file = os.path.join(output_folder, collection_file)
            lines_to_write = []
            with open(os.path.join(collection_folder, collection_file), "r") as f:
                with open(output_file, "w") as fw:
                    for line in f:
                        data = json.loads(line)
                        id = data["id"]
                        doc = data["contents"]
                        doc_tokens = analyzer.analyze(doc)
                        lines_to_write.append(json.dumps({"id": id, "contents": " ".join(doc_tokens)}) + "\n")
                        if len(lines_to_write) == 1000:
                            fw.writelines(lines_to_write)
                            lines_to_write = []
                    if len(lines_to_write) > 0:
                        fw.writelines(lines_to_write)
        return None
    if model == "krovetz":
        analyzer = Analyzer(get_lucene_analyzer(stemmer="krovetz", stopwords=False))
        collection_folder = f"{collection_folder}"
        output_folder = f"{collection_folder}_krovetz"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for collection_file in tqdm(collection_files):
            output_file = os.path.join(output_folder, collection_file)
            lines_to_write = []
            with open(os.path.join(collection_folder, collection_file), "r") as f:
                with open(output_file, "w") as fw:
                    for line in f:
                        data = json.loads(line)
                        id = data["id"]
                        doc = data["contents"]
                        doc_tokens = analyzer.analyze(doc)
                        lines_to_write.append(json.dumps({"id": id, "contents": " ".join(doc_tokens)}) + "\n")
                        if len(lines_to_write) == 1000:
                            fw.writelines(lines_to_write)
                            lines_to_write = []
                    if len(lines_to_write) > 0:
                        fw.writelines(lines_to_write)
        return None

    output_folder = f"{collection_folder}_{type}_stemmed_{arg.model}"

    if type=="vocab":

        mapping = os.path.join("output", f"stemmed_{type}_" + model + "_" + arg.dataset + ".jsonl")
        analyzer = Analyzer(get_lucene_analyzer(stemming=False, stopwords=False))

        with open(mapping, "r") as f:
            mapping_dict = json.load(f)
        print(len(mapping_dict))
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        for collection_file in tqdm(collection_files):
            output_file = os.path.join(output_folder, collection_file)
            lines_to_write = []
            with open(os.path.join(collection_folder, collection_file), "r") as f:
                with open(output_file, "w") as fw:
                    for line in tqdm(f):
                        data = json.loads(line)
                        id = data["id"]
                        doc = data["contents"]
                        doc_tokens = analyzer.analyze(doc)
                        new_doc_tokens = []
                        for token in doc_tokens:
                            if token in mapping_dict:
                                new_doc_tokens.extend(mapping_dict[token])
                            else:
                                new_doc_tokens.append(token)

                        lines_to_write.append(json.dumps({"id": id, "contents": " ".join(new_doc_tokens)}) + "\n")
                        if len(lines_to_write) == 1000:
                            fw.writelines(lines_to_write)
                            lines_to_write = []
                    if len(lines_to_write) > 0:
                        fw.writelines(lines_to_write)
    else:
        analyzer = Analyzer(get_lucene_analyzer(stemmer="porter", stopwords=False))
        #analyser2 = Analyzer(get_lucene_analyzer(stemming=False, stopwords=False))
        mapping_folder = f"out_collections/output_{model}_{collection_folder}_{type}"
        mapping_files = [f for f in os.listdir(mapping_folder) if f.endswith(".json")]
        mapping_dict = {}
        collection_counter = 0
        if type == "context":
            for mapping_file in mapping_files:
                with open(os.path.join(mapping_folder, mapping_file), "r") as f:
                    for line in f:
                        data = json.loads(line)
                        new_contents = data["contents"]
                        mapping_dict[data["id"]] = new_contents
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            for collection_file in tqdm(collection_files):
                output_file = os.path.join(output_folder, collection_file)
                lines_to_write = []
                with open(os.path.join(collection_folder, collection_file), "r") as f:
                    with open(output_file, "w") as fw:
                        for line in f:
                            data = json.loads(line)
                            id = data["id"]
                            doc = data["contents"].lower()
                            new_doc = doc
                            if id in mapping_dict:
                                new_doc = mapping_dict[id]
                            lines_to_write.append(json.dumps({"id": id, "contents": new_doc}) + "\n")
                            if len(lines_to_write) == 1000:
                                fw.writelines(lines_to_write)
                                lines_to_write = []
                        if len(lines_to_write) > 0:
                            fw.writelines(lines_to_write)
        elif type=="entities":
            output_folder = f"processed_collections/{collection_folder}_{type}_stemmed_{arg.model}_{vocab_model}_{type2}"
            if vocab_model!="porter":
                analyzer = Analyzer(get_lucene_analyzer(stemming=False, stopwords=False))
                vocab_file = os.path.join("output", f"stemmed_vocab_" + vocab_model + "_" + arg.dataset + ".jsonl")
                with open(vocab_file, "r") as f:
                    vocab_dict = json.load(f)


            for mapping_file in mapping_files:
                with open(os.path.join(mapping_folder, mapping_file), "r") as f:
                    for line in f:
                        data = json.loads(line)
                        entities = data["entities"]
                        processed_entities = process_entities(entities)
                        #make sure processed_entities is not ""

                        mapping_dict[data["id"]] = processed_entities
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            for collection_file in tqdm(collection_files):
                output_file = os.path.join(output_folder, collection_file)
                lines_to_write = []
                with open(os.path.join(collection_folder, collection_file), "r") as f:
                    with open(output_file, "w") as fw:
                        for line in f:
                            data = json.loads(line)
                            id = data["id"]
                            doc = data["contents"].lower()
                            current_mapping = []
                            if id in mapping_dict:
                                current_mapping = mapping_dict[id]
                            doc_tokens = []
                            counter = 0
                            doc_tokens = analyzer.analyze(doc)

                            current_mapping = sorted(current_mapping, key=len, reverse=True)

                            for entity in current_mapping:
                                count = doc.count(entity)
                                if count>0:
                                    doc_tokens.extend([entity] * count)
                                    if type2=="replace":
                                        doc = doc.replace(entity, "")


                                # count = doc.count(entity)
                                # if count>0:
                                #     counter += 1
                                #     doc_tokens.extend([entity]*count)
                                    #doc = doc.replace(entity, "")
                            #print(doc)
                            #exit(1)
                            if counter == 0:
                                # then reserve these entities as they might be incorrectly mapped
                                #print(collection_file, id, current_mapping)
                                collection_counter += 1

                            if vocab_model=="porter":
                                doc_tokens = doc_tokens
                            else:
                                tokens = analyzer.analyze(doc)
                                for token in tokens:
                                    if token in vocab_dict:
                                        doc_tokens.extend(vocab_dict[token])
                                    else:
                                        doc_tokens.append(token)
                            lines_to_write.append(json.dumps({"id": id, "contents": " ".join(doc_tokens)}) + "\n")
                            if len(lines_to_write) == 1000:
                                fw.writelines(lines_to_write)
                                lines_to_write = []
                        if len(lines_to_write) > 0:
                            fw.writelines(lines_to_write)
            print(collection_counter)





if __name__ == "__main__":
    main()

