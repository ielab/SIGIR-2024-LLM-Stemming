import openai
import torch
import argparse
from pyserini.index.lucene import IndexReader
import json
import os
#from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, AutoTokenizer, AutoModelForCausalLM,AutoModelForSeq2SeqLM,T5Tokenizer, T5ForConditionalGeneration, GenerationConfig
from tqdm import tqdm
import tiktoken
from multiprocessing import Pool
from functools import partial
import wandb

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BOS, EOS = "<s>", "</s>"
terms_sample = "programming\nprogrammer\nprograms”\nprograms.a\nPrograms PTY. LTD."
stemmed_sample = "programming: program\nprogrammer: program\nprograms: program\nprograms.a: program a\nPrograms PTY. LTD.: Programs PTY. LTD."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

openai.api_key = ""

openai_prompt = "You are a professional stemmer that is responsible to stem text. Text stemming is a natural language processing technique that is used to reduce words to their base form, also known as the root form. The process of stemming is used to normalize text and make it easier to process.\n" \
                "Your output should strictly follow the format 'original word:stem'. If a single original word produces multiple stems, separate the stems with a space.\n" \
                "Example: Can you provide the stemmed version of these terms?\n" \
                f"{terms_sample}\nStemmer:\n{stemmed_sample}\n" \
                "Can you provide the stemmed version of these terms?\n{terms}\n" \
                "Stemmer:"


llama_prompt = f"{BOS}{B_INST} {B_SYS}\n" \
                "You are a professional stemmer that is responsible to stem text. Text stemming is a natural language processing technique that is used to reduce words to their base form, also known as the root form. The process of stemming is used to normalize text and make it easier to process.\n" \
                "Your output should strictly follow the format 'original word:stem'. If a single original word produces multiple stems, separate the stems with a space.\n\n" \
                f"{E_SYS}\n\n" \
                f"Can you provide the stemmed version of these terms?\n{terms_sample} " \
                f"{E_INST} Stemmer:\n{stemmed_sample} {EOS}{BOS}{B_INST} " \
                "Can you provide the stemmed version of these terms?\n{terms} " \
                f"{E_INST} Stemmer:\n" \


def chunk_list(input_list, chunk_size):
    """Split the input list into chunks of size chunk_size."""
    return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]

def get_openai_response(prompt, out_token_count):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[{"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=4096-out_token_count-100,
    )
    context = response["choices"][0]["message"]["content"]
    return context

def get_llama_batch_responses(prompts, model, tokenizer, chunks):
    # Tokenize the batch of prompts
    #newline_token_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
    #print(prompts[0])
    encoded_inputs = tokenizer.batch_encode_plus(
        prompts,
        truncation=True,  # Truncate to model's max length
        padding='longest',
        return_tensors='pt',
        max_length=1024,
    )
    #print normal token length

    print("input_ids", encoded_inputs["input_ids"].shape)

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
    print(response_sentences)

    #print(response_sentences[0])
    final_dict = {}
    #split and take after Stemmed paragraph:
    for i, response in enumerate(response_sentences):
        if E_INST in response:
            current_response = response.split("Stemmer:")[2].strip().split("\n")
            current_chunk = chunks[i]
            out_dict = response_processing(current_chunk, current_response)
            final_dict.update(out_dict)

    return final_dict


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




def worker(model, chunk):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    current_chunk = [str(term) for term in chunk]
    chunk = "\n".join(current_chunk)
    prompt = openai_prompt.format(terms=chunk)
    token_count = len(encoding.encode(prompt))
    try:
        response = get_openai_response(prompt, token_count)
    except:
        try:
            response = get_openai_response(prompt, token_count)
        except:
            print("Error in getting response")

            return None
    responses = response.split("\n")
    print(responses)
    try:
        response_dict = response_processing(current_chunk, responses)
    except:
        print("Error in processing response")
        return None
    return response_dict


def response_processing(chunks, responses):
    response_dict = {}

    for i, item in enumerate(responses):
        if i >= len(chunks):
            break

        original_term = chunks[i]
        stemmed_terms = []

        # Split on '->' first, as this separator is more unique than ':'
        if '->' in item:
            parts = item.split('->', 1)
            if len(parts) == 2:
                potential_original, potential_stemmed = parts[0].strip(), parts[1].strip()
                if potential_original in original_term:
                    original_term = potential_original
                    stemmed_terms = potential_stemmed.split()

        # If still no stemmed_terms found, check the original term with a colon

        if original_term + ":" in item:
            potential_stemmed = item.split(original_term + ":")[1].strip()
            stemmed_terms = potential_stemmed.split()

        if not stemmed_terms and ':' in item:
            parts = item.split(':', 1)
            if len(parts) == 2:
                potential_original, potential_stemmed = parts[0].strip(), parts[1].strip()
                # Ensure that the original term fully matches, to avoid cases like "nyse:brk"
                if potential_original == original_term:
                    stemmed_terms = potential_stemmed.split()

        # If stemmed_terms are found, add them to the dictionary
        if stemmed_terms:
            response_dict[original_term] = stemmed_terms

    return response_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="robust04")
    parser.add_argument("--model", type=str, default="openai")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--batch", type=int, default=5)
    arg = parser.parse_args()
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project="llm-stem",

        # track hyperparameters and run metadata
        config={
            "dataset": arg.dataset,
            "model": arg.model,
        }
    )


    #dataset = ir_datasets.load(arg.dataset_name)
    #print(dataset)
    try:
        index_reader = IndexReader(arg.dataset)
    except:

        index_reader = IndexReader.from_prebuilt_index(arg.dataset)

    print(index_reader.stats())
    # get all the terms from the index
    terms = index_reader.terms()
    term_chunks = []
    print("Getting terms")

    output_dict = {}
    model_path = arg.model_path
    model = arg.model
    out_folder = "output"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    output_file = f'{out_folder}/stemmed_vocab_{model}.jsonl'
    if model=="llama":
        from transformers import AutoModel, LlamaForCausalLM, LlamaTokenizer, GenerationConfig, AutoTokenizer, \
            AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, GenerationConfig
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model_loaded = LlamaForCausalLM.from_pretrained(model_path).to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    #else:
        #tokenizer = tiktoken.tokenizer_for_model("gpt-3.5-turbo")



    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                current_dict = json.loads(line)
                output_dict.update(current_dict)

    for term in tqdm(terms):
        #make sure term is not pure number, or a list of numbers with punctuations
        current_term = term.term
        if is_number_or_number_list(current_term):
            continue
        #if term.cf <=1:
            #continue
        if current_term in output_dict:
            output_dict[current_term] = check_correct(current_term, output_dict[current_term])
            if len(output_dict[current_term]) ==0:
                term_chunks.append(term.term)
            continue

        term_chunks.append(term.term)
    print("Done getting terms")
    print(f"Total number of terms: {len(term_chunks)}")


    if model == "llama":
        chunk_size = 250
        term_chunks = chunk_list(term_chunks, chunk_size)
    else:
        chunk_size = 250
        term_chunks = chunk_list(term_chunks, chunk_size)

    print(f"Split terms into {len(term_chunks)} chunks")
    if model == "openai":
        with open(output_file, "a+") as fw:
            with Pool(processes=15) as pool:
                func = partial(worker, model)
                for response_dict in tqdm(pool.imap_unordered(func, term_chunks), total=len(term_chunks)):
                    if response_dict:
                        output_dict.update(response_dict)
                        fw.write(json.dumps(response_dict) + "\n")
    elif model == "llama":
        with open(output_file, "a+") as fw:
            for chunk in tqdm(term_chunks):
                batch_chunks_tem = chunk_list(chunk, int(chunk_size/arg.batch))
                batch_chunks = ["\n".join(batch_chunk) for batch_chunk in batch_chunks_tem]
                prompts = [llama_prompt.format(terms=batch_chunk) for batch_chunk in batch_chunks]

                try:
                    response_dict = get_llama_batch_responses(prompts, model_loaded, tokenizer, batch_chunks_tem)
                    fw.write(json.dumps(response_dict) + "\n")
                    output_dict.update(response_dict)
                except:
                    print("Error in getting response")
                    continue


if __name__ == "__main__":
    main()
