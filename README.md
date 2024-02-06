# SIGIR-2024-LLM-Stemming
Code Repo for LLM-stemming SIGIR-2024

### Enviroment Setup
1. Use Conda enviroment with python=3.10
2. Install all required package from [pyserini](https://github.com/castorini/pyserini/blob/master/docs/installation.md)
3. Install all required package from requirements.txt by running `pip install -r requirements.txt`

### Getting Data
```
python3 get_trec_data.py
python3 get_robust04.py
```


### Vocabulary Stemming (vs)

1. Index the original dataset without stemming, this can be done in pyserini, example on Trec_covid
```
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input trec_covid/ \
  --index indexes/trec_covid \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw
```
2. Run the vocabulary stemming script (remember to set openai api key in the script)
```
python3 stem-vocab.py --dataset indexes/trec_covid  \
    --model openai
 
python3 stem-vocab.py --dataset indexes/trec_covid  \
    --model llama \
    --model_path $LLAMA_MODEL_PATH
```

3. Reindex by rebuilding the collection with the new stemmer (only use openai to show, you can check the script in [trec_covid](scripts-trec-covid.sh), [robust04](scripts-robust04.sh) to get all the commands related)
```
python3 clean_vocab_mapping.py --dataset trec_covid --model openai
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model openai --type vocab
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model openai --type vocab


python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_vocab_stemmed_openai/ \
  --index indexes/trec_covid_vocab_stemmed_openai/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

```

4. Search using the new index and evaluate
```
python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_vocab_stemmed_openai/ \
  --topics topics/trec_covid/openai_vocab.tsv \
  --output runs/trec_covid/openai_vocab_stemmed.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/openai_vocab_stemmed.tsv
```


### Context Stemming (cs)

1. Run the context stemming script (no need to index the original dataset)
```
python3 stem-collection-context.py --dataset indexes/trec_covid  \
    --model llama \
    --model_path $LLAMA_MODEL_PATH \
    --batch $batch_size
    

python3 stem-queries-context.py --dataset indexes/trec_covid  \
    --model llama \
    --model_path $LLAMA_MODEL_PATH \
    --batch $batch_size  
    
```

2. Rebuild collection and queires, index and search (Below code can be found in [trec_covid](scripts-trec-covid.sh))
```
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model llama --type context
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model llama --type context


python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input trec_covid_context_stemmed_llama/ \
  --index indexes/trec_covid_context_stemmed_llama/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw


python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_context_stemmed_llama/ \
  --topics topics/trec_covid/llama_context.tsv \
  --output runs/trec_covid/llama_context.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/llama_context.tsv

```


### Entity-based Contextual Stemmings (ecs)
1. Run the context stemming script (no need to index the original dataset)
```
python3 stem-collection-entities.py --dataset indexes/trec_covid  \
    --model llama \
    --model_path $LLAMA_MODEL_PATH \
    --batch $batch_size
    

python3 stem-queries-entities.py --dataset indexes/trec_covid  \
    --model llama \
    --model_path $LLAMA_MODEL_PATH \
    --batch $batch_size  
    
```

2. Rebuild collection and queires, index and search (Below code can be found in [trec_covid](scripts-trec-covid.sh).)

**Note: For type_2: replace is ECS.1, append is ECS.2**
```
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model llama --type entities --type_2 replace
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model llama --type entities --type_2 replace

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_entities_stemmed_llama_porter_replace/ \
  --index indexes/trec_covid_entities_stemmed_llama_porter_replace/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_entities_stemmed_llama_porter_replace/ \
  --topics topics/trec_covid/llama_porter_replace_entities.tsv \
  --output runs/trec_covid/llama_entities_porter_replace.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/llama_entities_porter_replace.tsv

```


### Samples used in the paper

1. VS:
```
terms_sample = "programming\nprogrammer\nprograms”\nprograms.a\nPrograms PTY. LTD."
stemmed_sample = "programming: program\nprogrammer: program\nprograms: program\nprograms.a: program a\nPrograms PTY. LTD.: Programs PTY. LTD."
```

2. CS:
```
paragraph_sample = "Juliette lives in New York and represents the 'Stylish Trends' brand. She visits cafes and chats with locals, gaining inspiration for designs.\n"
stemmed_sample = "Juliette live in New York and represent the 'Stylish Trends' brand. She visit cafe and chat with local, gain inspir for design.\n"
```

3. ECS:
```
paragraph_sample = "Concurrent chilblains and retinal vasculitis in a child with COVID-19 Reports on COVID-19 in children are limited. Despite new data emerging, and understanding of the disease improves rapidly, there are as yet several features and complications related to the disease that remain unknown. Herein, we report the first case of a child with chilblains and retinal vasculitis related to COVID-19.\n"
sample_entities = "chilblains\nretinal vasculitis\nCOVID-19\n"
```




