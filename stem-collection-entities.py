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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

openai.api_key = "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"

#newline_token_id = None

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"

paragraph_sample = "Concurrent chilblains and retinal vasculitis in a child with COVID-19 Reports on COVID-19 in children are limited. Despite new data emerging, and understanding of the disease improves rapidly, there are as yet several features and complications related to the disease that remain unknown. Herein, we report the first case of a child with chilblains and retinal vasculitis related to COVID-19.\n"
sample_entities = "chilblains\nretinal vasculitis\nCOVID-19\n"

assistant_prompt = "You specialize in identifying and preserving entities, such as names, brands, or organisations, within text paragraphs. It's imperative to ensure these terms are not stemmed to enhance search engine performance."


openai_prompt = assistant_prompt + f"Example: Can you extract all entities from the following paragraph?\n{paragraph_sample}\nExtracted entities:\n{sample_entities}\n" \
                                          "Can you extract all entities from the following paragraph?\n{paragraph}\nExtracted entities:"


llama_prompt = f"{BOS}{B_INST} {B_SYS}\n" \
                f"{assistant_prompt}\n" \
                f"{E_SYS}\n\n" \
                f"Can you extract all entities from the following paragraph?\n{paragraph_sample} " \
                f"{E_INST} Extracted entities:\n{sample_entities} {EOS}{BOS}{B_INST} " \
                "Can you extract all entities from the following paragraph?\n{paragraph}\n " \
                f"{E_INST} Extracted entities:\n" \


def chunk_list(input_list, chunk_size):
    """Split the input list into chunks of size chunk_size."""
    return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]


def get_openai_response(prompt, out_token_count):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=2048-out_token_count-100,
    )
    context = response["choices"][0]["message"]["content"]
    return context


def get_llama_batch_responses(prompts, model, tokenizer, tem_ids):
    # Tokenize the batch of prompts
    #newline_token_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
    #print(prompts[0])
    encoded_inputs = tokenizer.batch_encode_plus(
        prompts,
        truncation=True,  # Truncate to model's max length
        padding='longest',
        return_tensors='pt',
        max_length=1280,
    )
    #print normal token length

    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)

    # Generate output for the batch

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        temperature=0.0000001,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        max_length=2048,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Decode the batch of output ids to sentenceswc -l
    response_sentences = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    output_sentences = []
    new_ids= []
    #split and take after Stemmed paragraph:
    for i, response in enumerate(response_sentences):
        try:
            if E_INST in response:
                new_ids.append(tem_ids[i])
                current_response = response.split("Extracted entities:")[2]
                if "\nNote:" in current_response:
                    current_response = current_response.split("\nNote:")[0]
                output_sentences.append(current_response.strip())
            else:
                print("Error in getting response")
                print(tem_ids[i])
        except:
            print("Error in getting response")
            print(tem_ids[i])
    return new_ids, output_sentences



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




def worker_openai(model, paragraph):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    prompt = openai_prompt.format(paragraph=paragraph["contents"])
    token_count = len(encoding.encode(prompt))

    try:
        response = get_openai_response(prompt, token_count)
    except:
        try:
            response = get_openai_response(prompt, token_count)
        except:
            print("Error in getting response")

            return None

    response_dict = {"id": paragraph["id"], "contents": paragraph["contents"], "entities":response.split("\n")}
    return response_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--model_path", type=str, default="llama")
    parser.add_argument("--batch", type=int, default=10)
    arg = parser.parse_args()

    model_path = arg.model_path
    input_file = arg.input_file
    os.environ["WANDB_API_KEY"] = "7e83983684637dd462cb826b6e6cc25cf29cbcab"
    wandb.init(
        # set the wandb project where this run will be logged
        project="llm-stem",

        # track hyperparameters and run metadata
        config={
            "dataset": input_file,
            "model": arg.model,
            "batch": arg.batch,
        }
    )


    output_dict = {}

    model = arg.model
    out_folder = f"output_{model}_{input_file.split('/')[0]}_entities"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    output_file = f'{out_folder}/{input_file.split("/")[-1]}'
    if model=="llama":
        from transformers import AutoModel, LlamaForCausalLM, LlamaTokenizer, GenerationConfig, AutoTokenizer, \
            AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, GenerationConfig
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model_loaded = LlamaForCausalLM.from_pretrained(model_path).to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    already_processe = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                current_dict = json.loads(line)
                already_processe.add(current_dict["id"])
    collection_list = []

    with open(input_file) as f:
        for line in f:
            current_dict = json.loads(line)
            if current_dict["id"] in already_processe:
                continue
            collection_list.append(current_dict)

    num = 0
    if model=="openai":
        if model == "openai":
            with open(output_file, "a+") as fw:
                with Pool(processes=1) as pool:
                    func = partial(worker_openai, model)
                    for response_dict in tqdm(pool.imap_unordered(func, collection_list), total=len(collection_list)):
                        if response_dict:
                            print("writing")
                            fw.write(json.dumps(response_dict) + "\n")
    else:
        chunked_collection_list = chunk_list(collection_list, arg.batch)
        for chunk in tqdm(chunked_collection_list):
            prompts = []
            tem_ids = []
            id_context_dict = {}
            for item in chunk:
                prompt = llama_prompt.format(paragraph=item["contents"])
                prompts.append(prompt)
                tem_ids.append(item["id"])
                id_context_dict[item["id"]] = item["contents"]
            new_ids, responses = get_llama_batch_responses(prompts, model_loaded, tokenizer, tem_ids)

            with open(output_file, "a+") as f:
                for id, response in zip(new_ids, responses):
                    response_dict = {"id": id, "contents": id_context_dict[id], "entities":response.split("\n")}
                    f.write(json.dumps(response_dict) + "\n")
    print(num)

if __name__ == "__main__":
    main()
