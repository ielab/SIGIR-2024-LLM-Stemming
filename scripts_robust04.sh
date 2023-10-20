
#default none
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input robust04/ \
  --index indexes/robust04 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  --stemmer none \
  -keepStopwords \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04 \
  --topics topics/robust04/default.tsv \
  --output runs/robust04/test-default.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

#python -m pyserini.eval.msmarco_passage_eval \
#   topics/robust04/default.tsv \
#   runs/robust04/test-default.tsv

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/test-default.tsv
------------------------------------------------------------------------------------------------------------------------

python3 remake_collection_based_on_mapping.py --dataset robust04 --model porter --type vocab
python3 remake_queries_based_on_mapping.py --dataset robust04 --model porter --type vocab

#porter
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input robust04_porter/ \
  --index indexes/robust04_porter \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_porter \
  --topics topics/robust04/porter.tsv \
  --output runs/robust04/test-porter.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128


trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/test-porter.tsv

------------------------------------------------------------------------------------------------------------------------

python3 remake_collection_based_on_mapping.py --dataset robust04 --model krovetz --type vocab
python3 remake_queries_based_on_mapping.py --dataset robust04 --model krovetz --type vocab
#krovetz
python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input robust04_krovetz/ \
  --index indexes/robust04_krovetz \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_krovetz \
  --topics topics/robust04/krovetz.tsv \
  --output runs/robust04/test-krovetz.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128


trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/test-krovetz.tsv

------------------------------------------------------------------------------------------------------------------------

#openai based
python3 clean_vocab_mapping.py --dataset robust04 --model openai
python3 remake_collection_based_on_mapping.py --dataset robust04 --model openai --type vocab
python3 remake_queries_based_on_mapping.py --dataset robust04 --model openai --type vocab


python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input robust04_vocab_stemmed_openai/ \
  --index indexes/robust04_vocab_stemmed_openai/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw


python3 -m pyserini.search.lucene \
  --index indexes/robust04_vocab_stemmed_openai/ \
  --topics topics/robust04/openai_vocab.tsv \
  --output runs/robust04/openai_vocab_stemmed.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/openai_vocab_stemmed.tsv
------------------------------------------------------------------------------------------------------------------------
#llama based

python3 remake_collection_based_on_mapping.py --dataset robust04 --model llama --type vocab
python3 remake_queries_based_on_mapping.py --dataset robust04 --model llama --type vocab


python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input robust04_vocab_stemmed_llama/ \
  --index indexes/robust04_vocab_stemmed_llama/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw


python3 -m pyserini.search.lucene \
  --index indexes/robust04_vocab_stemmed_llama/ \
  --topics topics/robust04/llama_vocab.tsv \
  --output runs/robust04/llama_vocab_stemmed.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/llama_vocab_stemmed.tsv


------------------------------------------------------------------------------------------------------------------------
#context based llama
python3 remake_collection_based_on_mapping.py --dataset robust04 --model llama --type context
python3 remake_queries_based_on_mapping.py --dataset robust04 --model llama --type context

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_context_stemmed_llama/ \
  --index indexes/robust04_context_stemmed_llama/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_context_stemmed_llama/ \
  --topics topics/robust04/llama_context.tsv \
  --output runs/robust04/llama_context.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/llama_context.tsv



------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab porter replace
python3 remake_collection_based_on_mapping.py --dataset robust04 --model llama --type entities --type_2 replace
python3 remake_queries_based_on_mapping.py --dataset robust04 --model llama --type entities --type_2 replace

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_entities_stemmed_llama_porter_replace/ \
  --index indexes/robust04_entities_stemmed_llama_porter_replace/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_entities_stemmed_llama_porter_replace/ \
  --topics topics/robust04/llama_porter_replace_entities.tsv \
  --output runs/robust04/llama_entities_porter_replace.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/llama_entities_porter_replace.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab porter append
python3 remake_collection_based_on_mapping.py --dataset robust04 --model llama --type entities --type_2 append
python3 remake_queries_based_on_mapping.py --dataset robust04 --model llama --type entities --type_2 append

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_entities_stemmed_llama_porter_append/ \
  --index indexes/robust04_entities_stemmed_llama_porter_append/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_entities_stemmed_llama_porter_append/ \
  --topics topics/robust04/llama_porter_append_entities.tsv \
  --output runs/robust04/llama_entities_porter_append.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/llama_entities_porter_append.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_openai replace
python3 remake_collection_based_on_mapping.py --dataset robust04 --model llama --type entities --vocab_model openai
python3 remake_queries_based_on_mapping.py --dataset robust04 --model llama --type entities --vocab_model openai

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_entities_stemmed_llama_openai_replace/ \
  --index indexes/robust04_entities_stemmed_llama_openai_replace/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_entities_stemmed_llama_openai_replace/ \
  --topics topics/robust04/llama_openai_replace_entities.tsv \
  --output runs/robust04/llama_entities_openai_replace.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/llama_entities_openai_replace.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_openai append
python3 remake_collection_based_on_mapping.py --dataset robust04 --model llama --type entities --vocab_model openai --type_2 append
python3 remake_queries_based_on_mapping.py --dataset robust04 --model llama --type entities --vocab_model openai --type_2 append

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_entities_stemmed_llama_openai_append/ \
  --index indexes/robust04_entities_stemmed_llama_openai_append/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_entities_stemmed_llama_openai_append/ \
  --topics topics/robust04/llama_openai_append_entities.tsv \
  --output runs/robust04/llama_entities_openai_append.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/llama_entities_openai_append.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_llama
python3 remake_collection_based_on_mapping.py --dataset robust04 --model llama --type entities --vocab_model llama
python3 remake_queries_based_on_mapping.py --dataset robust04 --model llama --type entities --vocab_model llama

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_entities_stemmed_llama_llama_replace/ \
  --index indexes/robust04_entities_stemmed_llama_llama_replace/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_entities_stemmed_llama_llama_replace/ \
  --topics topics/robust04/llama_llama_replace_entities.tsv \
  --output runs/robust04/llama_entities_llama_replace.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/llama_entities_llama_replace.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_llama append
python3 remake_collection_based_on_mapping.py --dataset robust04 --model llama --type entities --vocab_model llama --type_2 append
python3 remake_queries_based_on_mapping.py --dataset robust04 --model llama --type entities --vocab_model llama --type_2 append

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_entities_stemmed_llama_llama_append/ \
  --index indexes/robust04_entities_stemmed_llama_llama_append/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_entities_stemmed_llama_llama_append/ \
  --topics topics/robust04/llama_llama_append_entities.tsv \
  --output runs/robust04/llama_entities_llama_append.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/llama_entities_llama_append.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based roberta with vocab_porter replace
python3 remake_collection_based_on_mapping.py --dataset robust04 --model roberta --type entities --type_2 replace
python3 remake_queries_based_on_mapping.py --dataset robust04 --model roberta --type entities --type_2 replace

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_entities_stemmed_roberta_porter_replace/ \
  --index indexes/robust04_entities_stemmed_roberta_porter_replace/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_entities_stemmed_roberta_porter_replace/ \
  --topics topics/robust04/roberta_porter_replace_entities.tsv \
  --output runs/robust04/roberta_entities_porter_replace.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/roberta_entities_porter_replace.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based roberta with vocab_porter append
python3 remake_collection_based_on_mapping.py --dataset robust04 --model roberta --type entities --type_2 append
python3 remake_queries_based_on_mapping.py --dataset robust04 --model roberta --type entities --type_2 append

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_entities_stemmed_roberta_porter_append/ \
  --index indexes/robust04_entities_stemmed_roberta_porter_append/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_entities_stemmed_roberta_porter_append/ \
  --topics topics/robust04/roberta_porter_append_entities.tsv \
  --output runs/robust04/roberta_entities_porter_append.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/roberta_entities_porter_append.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based roberta with openai stemmer replace
python3 remake_collection_based_on_mapping.py --dataset robust04 --model roberta --type entities --vocab_model openai --type_2 replace
python3 remake_queries_based_on_mapping.py --dataset robust04 --model roberta --type entities --vocab_model openai --type_2 replace

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_entities_stemmed_roberta_openai_replace/ \
  --index indexes/robust04_entities_stemmed_roberta_openai_replace/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_entities_stemmed_roberta_openai_replace \
  --topics topics/robust04/roberta_openai_replace_entities.tsv \
  --output runs/robust04/roberta_entities_openai_replace.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/roberta_entities_openai_replace.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based roberta with openai stemmer append
python3 remake_collection_based_on_mapping.py --dataset robust04 --model roberta --type entities --vocab_model openai --type_2 append
python3 remake_queries_based_on_mapping.py --dataset robust04 --model roberta --type entities --vocab_model openai --type_2 append

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_entities_stemmed_roberta_openai_append/ \
  --index indexes/robust04_entities_stemmed_roberta_openai_append/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_entities_stemmed_roberta_openai_append \
  --topics topics/robust04/roberta_openai_append_entities.tsv \
  --output runs/robust04/roberta_entities_openai_append.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/roberta_entities_openai_append.tsv
------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_llama replace
python3 remake_collection_based_on_mapping.py --dataset robust04 --model roberta --type entities --vocab_model llama --type_2 replace
python3 remake_queries_based_on_mapping.py --dataset robust04 --model roberta --type entities --vocab_model llama --type_2 replace

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_entities_stemmed_roberta_llama_replace/ \
  --index indexes/robust04_entities_stemmed_roberta_llama_replace/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_entities_stemmed_roberta_llama_replace/ \
  --topics topics/robust04/roberta_llama_replace_entities.tsv \
  --output runs/robust04/roberta_entities_llama_replace.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/roberta_entities_llama_replace.tsv

------------------------------------------------------------------------------------------------------------------------
#entities based llama with vocab_llama append
python3 remake_collection_based_on_mapping.py --dataset robust04 --model roberta --type entities --vocab_model llama --type_2 append
python3 remake_queries_based_on_mapping.py --dataset robust04 --model roberta --type entities --vocab_model llama --type_2 append

python3 -m pyserini.index.lucene \
  --collection JsonCollection \
  --input processed_collections/robust04_entities_stemmed_roberta_llama_append/ \
  --index indexes/robust04_entities_stemmed_roberta_llama_append/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 20 \
  --optimize \
  -keepStopwords \
  --stemmer none \
  --storePositions --storeRaw

python3 -m pyserini.search.lucene \
  --index indexes/robust04_entities_stemmed_roberta_llama_append/ \
  --topics topics/robust04/roberta_llama_append_entities.tsv \
  --output runs/robust04/roberta_entities_llama_append.tsv \
  --output-format trec \
  --hits 1000 \
  --bm25 --k1 0.82 --b 0.68 \
  --threads 30 --batch-size 128

trec_eval -m recip_rank -m ndcg_cut.10 -m recall.1000 topics/robust04/qrels.txt runs/robust04/roberta_entities_llama_append.tsv