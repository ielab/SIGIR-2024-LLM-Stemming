from pyserini.index.lucene import IndexReader
from pyserini.search.lucene import LuceneSearcher
import json
from tqdm import tqdm
import os

#parameters to preset
output_folder = "trec_covid"
chunk = 34200
num_token = 300 #num_tokens set to 300 in the experiments


def chunk_list(input_list, chunk_size):
    """Split the input list into chunks of size chunk_size."""
    return [input_list[i:i+chunk_size] for i in range(0, len(input_list), chunk_size)]


searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-trec-covid.flat')
index = IndexReader.from_prebuilt_index('beir-v1.0.0-trec-covid.flat')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

stats = index.stats()

print(stats)
overall_list = []
# Iterate through all documents
counter = 0
for i in tqdm(range(0, searcher.num_docs)):
    docid = searcher.doc(i).get('id')
    current_doc = json.loads(searcher.doc(docid).raw())
    contents = current_doc["title"] + " " + current_doc["text"]

    #get only the top 768 terms maximum.

    contents = contents.split(" ")
    if len(contents) > num_token:
        counter +=1
        #print(docid, len(contents))
        contents = contents[:num_token]
    contents = " ".join(contents)

    new_dict = {"id": current_doc["_id"],
                "contents": contents}
    overall_list.append(new_dict)

print(counter)


overall_list = chunk_list(overall_list, chunk)
for chunk in overall_list:
    chunk.sort(key=lambda x: len(x["contents"].split()), reverse=True)


for i, chunk in enumerate(overall_list):
    output_path = os.path.join(output_folder, 'docs{:02d}.json'.format(i))
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        for doc in chunk:
            f.write(json.dumps(doc) + '\n')
    print(f"Done writing chunk {i}")


