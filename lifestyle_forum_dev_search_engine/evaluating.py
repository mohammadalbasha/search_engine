import searchService as search
import ir_datasets
lifestyleForum_dataset = ir_datasets.load("lotte/lifestyle/dev/forum")



################################################################################
########################### QRELS ####################################
################################################################################

qrel_dict = {}
real_relevant = {}
for qrel in lifestyleForum_dataset.qrels_iter():
    # print(qrel) # namedtuple<query_id, doc_id, relevance, iteration>
    query_id = qrel[0]
    doc_id = qrel[1]
    relevance = qrel[2]
    if query_id not in qrel_dict:
            qrel_dict[query_id] = {}
    qrel_dict[query_id][doc_id] = relevance
    if relevance > 0:
        if query_id not in real_relevant:
            real_relevant[query_id] = []
        real_relevant[query_id].append(doc_id)




################################################################################
########################### Retrieved documents for each query ####################################
################################################################################


# get the retrieved docs for each query by our search engine
retrieved_docs = {}
for query in lifestyleForum_dataset.queries_iter():
    #print(query) # namedtuple<query_id, text>
    ret_docs = search.docsIdsSearch(query.text)
    retrieved_docs[query.query_id] = ret_docs


################################################################################
########################### MAP CALCULATING ####################################
################################################################################

# Calculate average precision for each query
ap_list = []
for query in lifestyleForum_dataset.queries_iter():
    relevant_count = 0
    precision_sum = 0.0
    retrieved_count = 0
    for j, doc in enumerate(retrieved_docs[query.query_id]):
        retrieved_count += 1
        #print(qrel_dict[query.query_id])
        #print(j, doc)

        if qrel_dict[query.query_id].get(doc, 0) == 1:
            relevant_count += 1
            precision_sum += relevant_count / (j+1)
        
    if retrieved_count > 0:
        ap = precision_sum / retrieved_count
        ap_list.append(ap)
    # this may be wrong    
    else: 
        ap_list.append(0)

#print (ap_list)
# Calculate mean average precision
map_score = sum(ap_list) / len(ap_list)
print("MAP:", map_score)




################################################################################
########################### MRR CALCULATING ####################################
################################################################################

# Calculate reciprocal rank for each query
rr_list = []
for query in lifestyleForum_dataset.queries_iter():
    for j, doc in enumerate(retrieved_docs[query.query_id]):
        
        if qrel_dict[query.query_id].get(doc, 0) == 1:
            rr_list.append(1 / (j+1))
            break

        # this may be wrong
        rr_list.append(0)

# Calculate mean reciprocal rank
mrr_score = sum(rr_list) / len(rr_list)
print("MRR:", mrr_score)




################################################################################
########################### mean precision ####################################
################################################################################



precision_list = []
for query in lifestyleForum_dataset.queries_iter():
    relevant_count = 0
    retrieved_count = 0
    for j, doc in enumerate(retrieved_docs[query.query_id]):
        retrieved_count += 1

        # 3 : It can be an answer to the question, however, it is notsufficiently convincing. There should be an answer with much better quality for the question.    
        # 4 : It looks reasonable and convincing. Its quality is on parwith or better than the "Possibly Correct Answer". Note that it does not have to provide the same answer as the "PossiblyCorrect Answer".

        # if qrel_dict[query.query_id].get(doc, 0) == 4:
        #     relevant_count += 1
        # if qrel_dict[query.query_id].get(doc, 0) == 3:
        #     relevant_count += 0.5

        if qrel_dict[query.query_id].get(doc, 0) > 0:
                relevant_count += 1

    if retrieved_count > 0:
        precision = relevant_count / retrieved_count
        precision_list.append(precision)
    
    # this may be wrong
    else: 
        precision_list.append(0)


# Calculate mean precision
precision = sum(precision_list) / len(precision_list)
print("mean Precision:", precision)


################################################################################
########################### mean recall ####################################
################################################################################



recall_list = []
for query in lifestyleForum_dataset.queries_iter():
    relevant_count = 0
    all_relevant_count = len(real_relevant[query.query_id])
    for j, doc in enumerate(retrieved_docs[query.query_id]):

        # 3 : It can be an answer to the question, however, it is notsufficiently convincing. There should be an answer with much better quality for the question.    
        # 4 : It looks reasonable and convincing. Its quality is on parwith or better than the "Possibly Correct Answer". Note that it does not have to provide the same answer as the "PossiblyCorrect Answer".

        # if qrel_dict[query.query_id].get(doc, 0) == 4:
        #     relevant_count += 1
        # if qrel_dict[query.query_id].get(doc, 0) == 3:
        #     relevant_count += 0.5

        if qrel_dict[query.query_id].get(doc, 0) > 0:
                relevant_count += 1

    if all_relevant_count > 0:
        recall = relevant_count / all_relevant_count
        recall_list.append(recall)
    
    # this may be wrong
    else: 
        recall_list.append(0)


# Calculate mean recall
recall = sum(recall_list) / len(recall_list)
print("mean recall:", recall)






################################################################################
########################### precision@k ####################################
################################################################################


k = 10
precision_list = []
for query in lifestyleForum_dataset.queries_iter():
    relevant_count = 0
    retrieved_count = 0
    for j, doc in enumerate(retrieved_docs[query.query_id]):
        if (retrieved_count == 10):
            break
        retrieved_count += 1


        if qrel_dict[query.query_id].get(doc, 0) == 1:
            relevant_count += 1

    if retrieved_count > 0:
        precision = relevant_count / retrieved_count
        precision_list.append(precision)
    
    # this may be wrong
    else: 
        precision_list.append(0)


# Calculate mean precision@k
precision_at_k = sum(precision_list) / len(precision_list)
print("Precision@k:", precision_at_k)