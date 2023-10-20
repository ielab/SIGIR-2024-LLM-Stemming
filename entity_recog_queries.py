import time
import openai
import torch
import argparse
import json
import os
from tqdm import tqdm
import tiktoken
from multiprocessing import Pool
from functools import partial
import wandb
from transformers import AutoTokenizer, AutoModelForTokenClassification


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_entities(texts, model, tokenizer, ids=None):
    """Get named entities from a list of texts using a pretrained NER model."""

    # Tokenize input texts
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        logits = model(**inputs).logits
    overall_entities = []
    # Decode logits to get entities
    predicted_indices = torch.argmax(logits, dim=2).cpu().numpy()

    for idx, input_ids in enumerate(inputs["input_ids"]):

        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        predictions = predicted_indices[idx]
        entities = []
        for token, label_idx in zip(tokens, predictions):
            label = model.config.id2label[label_idx]
            if label != "O":
                entities.append({"word": token, "entity": label})
        merged_entities = merge_tokens([entities])
        words = [entity["word"] for entity in merged_entities]
        overall_entities.append(words)

    return ids, overall_entities


def merge_tokens(token_entities):
    merged_results = []

    for entry in token_entities:
        current_word = ""
        current_entity = None
        current_start = None
        current_end = None
        last_merged_word = None

        for token in entry:
            if token["word"] == "<pad>":
                continue

            word = token["word"].replace("▁", "")

            # If we encounter a new entity or the token is a new word
            if current_entity != token["entity"] or token["word"].startswith("▁"):
                # If there was a previous entity and it's not a repeat of the last, save it
                if current_entity is not None and current_word != last_merged_word:
                    merged_results.append({
                        "word": current_word,
                        "entity": current_entity,
                        "start": current_start,
                        "end": current_end
                    })
                    last_merged_word = current_word

                # Start a new word
                current_entity = token["entity"]
                current_word = word
                current_start = token.get("start", None)
                current_end = token.get("end", None)

            # If it's the same entity and not a new word, append to the current word
            elif current_entity == token["entity"]:
                current_word += word
                current_end = token.get("end", None)

        # Append the last entity if there's one left and it's not a repeat
        if current_entity is not None and current_word != last_merged_word:
            merged_results.append({
                "word": current_word,
                "entity": current_entity,
                "start": current_start,
                "end": current_end
            })

    return merged_results


def chunk_list(input_list, chunk_size):
    """Split the input list into chunks of size chunk_size."""
    return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--model_path", type=str, default="51la5/roberta-large-NER")
    parser.add_argument("--batch", type=int, default=10)
    arg = parser.parse_args()


    model_path = arg.model_path
    input_file = arg.input_file
    model = "roberta"
    # wandb.login()
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="llm-stem",
    #
    #     # track hyperparameters and run metadata
    #     config={
    #         "dataset": input_file,
    #         "model": arg.model,
    #         "batch": arg.batch,
    #     }
    # )

    #openai.api_key = openai_keys[arg.key]
    #output_dict = {}



    output_file = os.path.join("/".join(input_file.split('/')[:-1]), f"{model}_entities.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_loaded = AutoModelForTokenClassification.from_pretrained(model_path).to(device)

    already_processe = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                current_dict = json.loads(line)
                already_processe.add(current_dict["id"])
    collection_list = []

    collection_list = []

    with open(input_file) as f:
        for line in f:
            try:
                qid, query = line.strip().split("\t")
            except:
                qid, query = line.strip().split(" ", 1)
                print(query)
            current_dict = {"id": qid, "contents": query}
            if qid in already_processe:
                continue
            collection_list.append(current_dict)
    #sort collection list based on the length of the query
    collection_list = sorted(collection_list, key=lambda x: len(x["contents"].split()), reverse=True)
    chunked_collection_list = chunk_list(collection_list, arg.batch)
    for chunk in tqdm(chunked_collection_list):
        passages = []
        tem_ids = []
        for item in chunk:
            passage = item["contents"]
            passages.append(passage)
            tem_ids.append(item["id"])
        new_ids, entities = get_entities(passages, model_loaded, tokenizer, tem_ids)

        with open(output_file, "a+") as f:
            for id, entity in zip(new_ids, entities):
                response_dict = {"id": id, "entities": entity}
                f.write(json.dumps(response_dict) + "\n")

if __name__ == "__main__":
    main()