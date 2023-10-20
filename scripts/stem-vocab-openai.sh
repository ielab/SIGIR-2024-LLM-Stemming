
for i in {1..10}
do
  python3 stem-vocab.py --dataset indexes/trec_covid/ --model openai
done

for i in {1..2}
do
  python3 stem-vocab.py --dataset indexes/robust04/ --model openai
done