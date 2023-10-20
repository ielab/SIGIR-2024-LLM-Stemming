
#default none
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

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid \
  --topics topics/trec_covid/default.tsv \
  --output runs/robust04/test-default.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/test-default.tsv
------------------------------------------------------------------------------------------------------------------------

python3 remake_collection_based_on_mapping.py --dataset trec_covid --model porter --type vocab
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model porter --type vocab

#porter
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_porter/ \
  --index indexes/trec_covid_porter \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_porter \
  --topics topics/trec_covid/default.tsv \
  --output runs/trec_covid/test-porter.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/test-porter.tsv

------------------------------------------------------------------------------------------------------------------------
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model krovetz --type vocab
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model krovetz --type vocab

#krovetz
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_krovetz/ \
  --index indexes/trec_covid_krovetz \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_krovetz \
  --topics topics/trec_covid/default.tsv \
  --output runs/trec_covid/test-krovetz.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128


trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/test-krovetz.tsv

------------------------------------------------------------------------------------------------------------------------

#openai vocab based
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


python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_vocab_stemmed_openai/ \
  --topics topics/trec_covid/openai_vocab.tsv \
  --output runs/trec_covid/openai_vocab_stemmed.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/openai_vocab_stemmed.tsv
------------------------------------------------------------------------------------------------------------------------

#llama vocab based

python3 remake_collection_based_on_mapping.py --dataset trec_covid --model llama --type vocab
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model llama --type vocab


python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input trec_covid_vocab_stemmed_llama/ \
  --index indexes/trec_covid_vocab_stemmed_llama/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw


python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_vocab_stemmed_llama/ \
  --topics topics/trec_covid/llama_vocab.tsv \
  --output runs/trec_covid/llama_vocab_stemmed.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/llama_vocab_stemmed.tsv

------------------------------------------------------------------------------------------------------------------------

#llama context based

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


------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_porter replace
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


------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_porter append
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model llama --type entities --type_2 append
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model llama --type entities --type_2 append

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_entities_stemmed_llama_porter_append/ \
  --index indexes/trec_covid_entities_stemmed_llama_porter_append/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_entities_stemmed_llama_porter_append/ \
  --topics topics/trec_covid/llama_porter_append_entities.tsv \
  --output runs/trec_covid/llama_entities_porter_append.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/llama_entities_porter_append.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_openai replace
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model llama --type entities --vocab_model openai
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model llama --type entities --vocab_model openai

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_entities_stemmed_llama_openai_replace/ \
  --index indexes/trec_covid_entities_stemmed_llama_openai_replace/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_entities_stemmed_llama_openai_replace/ \
  --topics topics/trec_covid/llama_openai_replace_entities.tsv \
  --output runs/trec_covid/llama_entities_openai_replace.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/llama_entities_openai_replace.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_openai append
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model llama --type entities --vocab_model openai --type_2 append
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model llama --type entities --vocab_model openai --type_2 append

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_entities_stemmed_llama_openai_append/ \
  --index indexes/trec_covid_entities_stemmed_llama_openai_append/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_entities_stemmed_llama_openai_append/ \
  --topics topics/trec_covid/llama_openai_append_entities.tsv \
  --output runs/trec_covid/llama_entities_openai_append.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/llama_entities_openai_append.tsv
------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_llama repalce
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model llama --type entities --vocab_model llama
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model llama --type entities --vocab_model llama

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_entities_stemmed_llama_llama_replace/ \
  --index indexes/trec_covid_entities_stemmed_llama_llama_replace/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_entities_stemmed_llama_llama_replace/ \
  --topics topics/trec_covid/llama_llama_replace_entities.tsv \
  --output runs/trec_covid/llama_entities_llama_replace.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/llama_entities_llama_replace.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_llama append
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model llama --type entities --vocab_model llama --type_2 append
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model llama --type entities --vocab_model llama --type_2 append

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_entities_stemmed_llama_llama_append/ \
  --index indexes/trec_covid_entities_stemmed_llama_llama_append/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_entities_stemmed_llama_llama_append/ \
  --topics topics/trec_covid/llama_llama_append_entities.tsv \
  --output runs/trec_covid/llama_entities_llama_append.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/llama_entities_llama_append.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based roberta with vocab_porter replace
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model roberta --type entities --type_2 replace
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model roberta --type entities --type_2 replace

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_entities_stemmed_roberta_porter_replace/ \
  --index indexes/trec_covid_entities_stemmed_roberta_porter_replace/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_entities_stemmed_roberta_porter_replace/ \
  --topics topics/trec_covid/roberta_porter_replace_entities.tsv \
  --output runs/trec_covid/roberta_entities_porter_replace.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/roberta_entities_porter_replace.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based roberta with vocab_porter append
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model roberta --type entities --type_2 append
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model roberta --type entities --type_2 append

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_entities_stemmed_roberta_porter_append/ \
  --index indexes/trec_covid_entities_stemmed_roberta_porter_append/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_entities_stemmed_roberta_porter_append/ \
  --topics topics/trec_covid/roberta_porter_append_entities.tsv \
  --output runs/trec_covid/roberta_entities_porter_append.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/roberta_entities_porter_append.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based roberta with openai stemmer replace
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model roberta --type entities --vocab_model openai
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model roberta --type entities --vocab_model openai

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_entities_stemmed_roberta_openai_replace/ \
  --index indexes/trec_covid_entities_stemmed_roberta_openai_replace/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_entities_stemmed_roberta_openai_replace/ \
  --topics topics/trec_covid/roberta_openai_replace_entities.tsv \
  --output runs/trec_covid/roberta_entities_openai_replace.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/roberta_entities_openai_replace.tsv


------------------------------------------------------------------------------------------------------------------------
#entities based roberta with openai stemmer append
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model roberta --type entities --vocab_model openai --type_2 append
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model roberta --type entities --vocab_model openai --type_2 append

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_entities_stemmed_roberta_openai_append/ \
  --index indexes/trec_covid_entities_stemmed_roberta_openai_append/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_entities_stemmed_roberta_openai_append/ \
  --topics topics/trec_covid/roberta_openai_append_entities.tsv \
  --output runs/trec_covid/roberta_entities_openai_append.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/roberta_entities_openai_append.tsv


------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_llama replace
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model roberta --type entities --vocab_model llama
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model roberta --type entities --vocab_model llama

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_entities_stemmed_roberta_llama_replace/ \
  --index indexes/trec_covid_entities_stemmed_roberta_llama_replace/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_entities_stemmed_roberta_llama_replace/ \
  --topics topics/trec_covid/roberta_llama_replace_entities.tsv \
  --output runs/trec_covid/roberta_entities_llama_replace.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/roberta_entities_llama_replace.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_llama append
python3 remake_collection_based_on_mapping.py --dataset trec_covid --model roberta --type entities --vocab_model llama --type_2 append
python3 remake_queries_based_on_mapping.py --dataset trec_covid --model roberta --type entities --vocab_model llama --type_2 append

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/trec_covid_entities_stemmed_roberta_llama_append/ \
  --index indexes/trec_covid_entities_stemmed_roberta_llama_append/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/trec_covid_entities_stemmed_roberta_llama_append/ \
  --topics topics/trec_covid/roberta_llama_append_entities.tsv \
  --output runs/trec_covid/roberta_entities_llama_append.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m map -m ndcg_cut.10 -m recall.1000 topics/trec_covid/qrels.txt runs/trec_covid/roberta_entities_llama_append.tsv



