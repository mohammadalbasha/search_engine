# import ir_datasets
# dataset = ir_datasets.load("beir/fever/dev")
# for query in dataset.docs_iter():
#     print(query.doc_id.split('_')[1]) # namedtuple<query_id, text>


# import ir_datasets
# dataset = ir_datasets.load("lotte/lifestyle/dev/forum")
# for query in dataset.queries_iter():
#     query # namedtuple<query_id, text>
    
import ir_datasets
dataset = ir_datasets.load("lotte/lifestyle/dev/forum")
for doc in dataset.docs_iter():
    print(doc)
     # namedtuple<query_id, text>