import json
import random
import torch
import os
import faiss
import numpy as np
from tqdm import tqdm
import argparse

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader


knowledge_reader = KnowledgeReader('./data', 'knowledge_with_absa.json')

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--device', type=str, default='cuda:0', help='Device for model execution')
parser.add_argument('--dataset_split', type=str, default='val', help='Dataset split')
parser.add_argument('--positive_sample_size', type=int, default=3, help='Number of positive samples')
parser.add_argument('--negative_sample_size', type=int, default=1, help='Number of negative samples')
parser.add_argument('--model_name', type=str, default='QuaSiKS_reviewer', help='Model name')
args = parser.parse_args()


device = torch.device(args.device)
dataset_split = args.dataset_split
positive_sample_size = args.positive_sample_size
negative_sample_size = args.negative_sample_size
model_name = args.model_name
if negative_sample_size != 0:
    #save_path = f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/{dataset_split}/QuaSiKS_reviewer_{positive_sample_size}pos{negative_sample_size}neg.json"
    save_path = f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/Dynamic_Dataset_copy/Resource_RG/QuaSiKS_reviewer_{positive_sample_size}pos{negative_sample_size}neg.json"
else:
    #save_path = f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/{dataset_split}/QuaSiKS_reviewer_{positive_sample_size}pos.json"
    save_path = f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/Dynamic_Dataset_copy/Resource_RG/QuaSiKS_reviewer_{positive_sample_size}pos.json"

print("device:",device,"dataset_split:",dataset_split,"positive_sample_size:",positive_sample_size,"negative_sample_size:",negative_sample_size)

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(f'/home/zhuangjt/zhuangjt_disk3/SK-TOD/results/{model_name}/checkpoint-78700')  # 指向保存 tokenizer 的目录
model = AutoModelForSeq2SeqLM.from_pretrained(f'/home/zhuangjt/zhuangjt_disk3/SK-TOD/results/{model_name}/checkpoint-78700')  # 指向保存模型的目录
model.to(device)

def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


em_knowledge = load_data(f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/{dataset_split}/em/em.json")
logs = load_data(f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/{dataset_split}/logs.json")
logs_aspect = load_data(f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/{dataset_split}/logs_aspect.json")
'''
em_knowledge = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/Dynamic_Dataset_copy/Resource_RG/em.json")
logs = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/Dynamic_Dataset_copy/Resource_RG/logs.json")
logs_aspect = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/Dynamic_Dataset_copy/Resource_RG/logs_aspect.json")
'''
vocab_dish_or_drink = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/vocab_dish_or_drink.json")

similarity_model = SentenceTransformer('/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/all-MiniLM-L6-v2')
##加载资源向量库
resourse_vector = np.load('/home/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/resourse/resource_vector_base.npy')
dimension = resourse_vector.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(resourse_vector)

resourse_labels = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/resourse/labels.json')
resourse_logs = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/resourse/logs.json')
resourse_logs_aspect = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/resourse/logs_aspect.json')
##先对resourse进行处理，构建三个列表，把所有label["target"] == false的排除掉
resourse_labels_new = []
resourse_logs_new = []
resourse_logs_aspect_new = []
for label,log,log_aspect in zip(resourse_labels,resourse_logs,resourse_logs_aspect):
    if label["target"] == True:
        resourse_labels_new.append(label)
        resourse_logs_new.append(log)
        resourse_logs_aspect_new.append(log_aspect)


def prepare_knowledge():
    knowledge_docs = get_snippet_list()
    knoweldege_snippets = {}
    for snippet_id, snippet in tqdm(enumerate(knowledge_docs), disable=False, desc='prepare knowledge snippets '):
            key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
            knoweldege_snippets[key] = snippet["doc"]["body"]
    return  knoweldege_snippets


def get_snippet_list():
    result = []
    for domain in knowledge_reader.get_domain_list():
        for entity_id in knowledge_reader.knowledge[domain].keys():
            for review_doc_id in knowledge_reader.get_review_doc_ids(domain, entity_id):
                review_doc = knowledge_reader.get_review_doc(domain, entity_id, review_doc_id)
                for review_sent_id, review_sent in review_doc['sentences'].items():
                    review_absa = knowledge_reader.get_review_absa(domain, entity_id, review_doc_id, review_sent_id)
                    result.append(
                        {'domain': domain, 'entity_id': entity_id, 'entity_name': review_doc['entity_name'],
                            'doc_id': f"{review_doc_id}-{review_sent_id}",
                            'doc': {'body': review_sent},
                            'absa': review_absa})
            for faq_doc_id in knowledge_reader.get_faq_doc_ids(domain, entity_id):
                faq_doc = knowledge_reader.get_faq_doc(domain, entity_id, faq_doc_id)
                faq_absa = knowledge_reader.get_faq_absa(domain, entity_id, faq_doc_id)
                #print("faq_absa:",faq_absa)
                result.append({'domain': domain, 'entity_id': entity_id, 'entity_name': faq_doc['entity_name'],
                                'doc_id': faq_doc_id,
                                'doc': {'body': f"{faq_doc['question']} {faq_doc['answer']}"},
                                'absa': faq_absa})
    return result


knowledge_snippets = prepare_knowledge() #key是domain__entity_id__doc_id,value是knowledge sentence


##few-shot构建函数
def create_few_shot(user_query, aspect_item_of_query, knowledge_sentence):
    
    ##这个函数主要是用于构建few-shot
    ##输入是用户的query
    ##然后根据用户的query，从resourse中找到最相似的query向量的索引
    ##并根据索引找到对应的query文本和aspect文本，knowledge标签字典
    ##从knowledge标签字典中随机选取两个作为positive,再到对应实体的知识列表中随机选择两个不在knowledge标签字典中的作为negative
    ##最后返回一个few-shot的列表，包含7个元素，分别是Definition,positive1,positive2,negative1,negative2,user_query,end_instruction
    definition = "Definition: The output will be an assessment of whether the knowledge can be used to support answering the user's query.If it can support, the output will be 'yes'; otherwise, the output will be 'no'."
    user_query_text = """Now complete the following example-
input:
[user_query]{}
[aspect_item_of_query]{}
[knowledge]{}
output:""".format(user_query,aspect_item_of_query,knowledge_sentence)
    
    ##计算query的向量
    query_embedding = similarity_model.encode([user_query])
    D, I = index.search(query_embedding, k=1)
    ##索引是I[0][0]
    socre = 1/(1 + D[0][0])
    #print(socre)
    resourse_labels = resourse_labels_new[I[0][0]]
    resourse_logs = resourse_logs_new[I[0][0]][-1]["text"]
    resourse_logs_aspect = resourse_logs_aspect_new[I[0][0]]

    
    #构建knowledge snippets（在最开始构建）,key是domain__entity_id__doc_id,value是knowledge sentence
    #在每一轮构建中，将正样本按照domain__entity_id__doc_id构建，获取相对应的knowledge sentence
    #同时，也要有一个正样本key_list,用于之后的随机抽取负样本的排除工作
    #获取knowledge_prefix_visited中的所有元素，然后从knowledge snippets中找到前缀是这些元素的key
    #然后将正样本的key_list中的元素从knowledge snippets中删除
    #然后从剩下的key中随机抽取两个作为负样本
    
    sample_knowledges = resourse_labels["knowledge"]
    positive_knowledges = []
    for knowledge in sample_knowledges:
        if knowledge["doc_type"] == "review":
            key = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}-{knowledge['sent_id']}"
        else:
            key = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}"
        positive_knowledges.append(key)

    knowledge_prefix_visited = set()
    knowledge_candidates = []
    for knowledge in sample_knowledges:
        prefix = "{}__{}".format(knowledge["domain"], knowledge["entity_id"])
        if prefix not in knowledge_prefix_visited:
            knowledge_prefix_visited.add(prefix)
            candidates = [
                cand 
                for cand in knowledge_snippets.keys()
                if "__".join(cand.split("__")[:-1]) == prefix
            ]
            knowledge_candidates.extend(candidates)
    
    #negative_knowledges是从knowledge_candidates中删除掉positive_knowledges的key_list中的元素之后的剩余元素中随机抽取两个
    negative_knowledges_key = []
    negative_knowledges_text = []

    if len(positive_knowledges) >=positive_sample_size:
        positive_knowledges_key = random.sample(positive_knowledges,positive_sample_size)
    else:
        positive_knowledges_key = positive_knowledges
    positive_knowledges_text = []
    for knowledge in positive_knowledges_key:
        positive_knowledges_text.append(knowledge_snippets[knowledge])
    
    positive_reviewers = set()
    for knowledge in positive_knowledges_key:
        if '-' in knowledge:
            positive_reviewers.add(knowledge.split('-')[0])
    if len(positive_reviewers) != 0:
        candidates_negative = []
        for positive_reviewer in positive_reviewers:
            candidates_negative.extend([candidate for candidate in knowledge_candidates if candidate.split('-')[0] == positive_reviewer])
        for knowledge in candidates_negative:
            if knowledge not in positive_knowledges:
                negative_knowledges_key.append(knowledge)        
    else:
        for knowledge in knowledge_candidates:
            if knowledge not in positive_knowledges:
                negative_knowledges_key.append(knowledge)
    
    if len(negative_knowledges_key) >= negative_sample_size:
        negative_knowledges_key = random.sample(negative_knowledges_key,negative_sample_size)
    else:
        negative_knowledges_key = negative_knowledges_key

    for knowledge in negative_knowledges_key:
        negative_knowledges_text.append(knowledge_snippets[knowledge])

    '''
    if len(negative_knowledges_text) >=2:
        negative_knowledges_text = random.sample(negative_knowledges_text,2)
    '''
    ##前面已经定义好Definition和Now complete the following example-之后的文本，所以只需要完成中间两个positive和negative的文本
    positive_text = ""
    
    for i in range(len(positive_knowledges_text)):
        positive_text += "Positive example {}-\ninput:\n[user_query]{}\n[aspect_item_of_query]{}\n[knowledge]{}\noutput:yes\n".format(i+1,resourse_logs,resourse_logs_aspect,positive_knowledges_text[i])
    
    if negative_sample_size != 0:
        negative_text = ""
        for i in range(len(negative_knowledges_text)):
            negative_text += "Negative example {}-\ninput:\n[user_query]{}\n[aspect_item_of_query]{}\n[knowledge]{}\noutput:no\n".format(i+1,resourse_logs,resourse_logs_aspect,negative_knowledges_text[i])
        input_text = definition + "\n" + positive_text + negative_text + user_query_text 
    else:
        input_text = definition + "\n" + positive_text + user_query_text 
    
    print(input_text)

    return input_text

def instructks(knowledge_sent, user_query, aspect_item_of_query):
    input_text = create_few_shot(user_query, aspect_item_of_query, knowledge_sent)
    
    input_encodings = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_encodings = input_encodings.to(device)
    output_sequences = model.generate(
        input_ids=input_encodings['input_ids'],
        attention_mask=input_encodings['attention_mask'],
        max_length=512,
    )
    output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    if output_text == "yes":
        return True
    else:
        return False


save_interval = 50  # Save after every 50 iterations


list = []
##加载emm_knowledge,logs,logs_aspect
for idx,(em, log, log_aspect) in enumerate(tqdm(zip(em_knowledge, logs, logs_aspect))):
    dict = {}
    if em["target"] == False:
        dict["target"] = False
        list.append(dict)
        continue
    else:
        dict["target"] = True
        user_query = log[-1]["text"]
        aspect_item_of_query = log_aspect
        
        ##获取实体的domain和entity_id
        knowledge_infs = em["knowledge"]
        knowledge_list = []
        for knowledge_info in knowledge_infs:  
            domain = knowledge_info["domain"]
            entity_id = knowledge_info["entity_id"]
            ##获取knowledge
            for review_doc_id in knowledge_reader.get_review_doc_ids(domain, entity_id):
                review_doc = knowledge_reader.get_review_doc(domain, entity_id, review_doc_id)
                for review_sent_id, review_sent in review_doc['sentences'].items():
                    knowledge_sent = review_sent
                    response = instructks(knowledge_sent, user_query, aspect_item_of_query)
                    if response == True:
                        dict_know = {}
                        dict_know["domain"] = domain
                        dict_know["entity_id"] = entity_id
                        dict_know["doc_type"] = "review"
                        dict_know["doc_id"] = int(review_doc_id)
                        dict_know["sent_id"] = int(review_sent_id)
                        knowledge_list.append(dict_know)
            for faq_doc_id in knowledge_reader.get_faq_doc_ids(domain, entity_id):
                faq_doc = knowledge_reader.get_faq_doc(domain, entity_id, faq_doc_id)
                knowledge_sent = f"question:{faq_doc['question']} answer:{faq_doc['answer']}"
                response = instructks(knowledge_sent, user_query, aspect_item_of_query)
                if response == True:
                    dict_know = {}
                    dict_know["domain"] = domain
                    dict_know["entity_id"] = entity_id
                    dict_know["doc_type"] = "faq"
                    dict_know["doc_id"] = int(faq_doc_id)
                    knowledge_list.append(dict_know)
        dict["knowledge"] = knowledge_list
        dict["response"] = "none"

    list.append(dict)

    #if (idx + 1) % save_interval == 0 or (idx + 1) == len(em_knowledge):
with open(save_path, 'w') as f:
    json.dump(list, f, indent=4)