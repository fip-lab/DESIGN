import openai
import os
import json
import pandas as pd
import numpy as np
import logging
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import random
from transformers import pipeline

os.environ["http_proxy"] = "http://localhost:7890"
os.environ["https_proxy"] = "http://localhost:7890"

my_key = "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"
my_org ='org-nedj4BwfkiH9r5kvEmOQBoIB'
openai.api_key = my_key
openai.organization = my_org

def get_gpt_response(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        result = response.choices[0].message["content"]
    except Exception as e:
        logging.info(e)
        result = "error"
        time.sleep(60)
    return result



dataset_type = "val"
top_k_shot = 3
sum_num = 6

def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def save_data(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def create_knowledge_sentence_for_entity(entity_knowledge_list,do_summary):
    knowledge_sentence = ""
    #aspect_sentiment_dict = {}
    doc_dict = {}
    for knowledge in entity_knowledge_list:
        if knowledge['doc_type'] == 'faq':
            question = knowledge_with_absa[knowledge["domain"]][str(knowledge["entity_id"])]["faqs"][str(knowledge["doc_id"])]["question"]
            answer = knowledge_with_absa[knowledge["domain"]][str(knowledge["entity_id"])]["faqs"][str(knowledge["doc_id"])]["answer"]
            knowledge_sentence += f"<faq> question: {question}  answer: {answer}\n"
        else:
            if f"{knowledge['domain']}_{knowledge['entity_id']}_{knowledge['doc_id']}" not in doc_dict.keys():
                doc_dict[f"{knowledge['domain']}_{knowledge['entity_id']}_{knowledge['doc_id']}"] = []
            doc_dict[f"{knowledge['domain']}_{knowledge['entity_id']}_{knowledge['doc_id']}"].append(knowledge)
    for doc in doc_dict.keys():
        doc_knowledge_list = doc_dict[doc]
        doc_knowledge_sentence = "<review> "
        for knowledge in doc_knowledge_list:
            domain = knowledge["domain"]
            entity_id = knowledge["entity_id"]
            doc_id = knowledge["doc_id"]
            sent_id = knowledge["sent_id"]
            doc_knowledge_sentence += f"{knowledge_with_absa[domain][str(entity_id)]['reviews'][str(doc_id)]['sentences'][str(sent_id)]}"
            aspect_sentiment = knowledge_with_absa[domain][str(entity_id)]["reviews"][str(doc_id)]["aspect_sentiment"][int(sent_id)]
            #aspect_sentiment_dict = {**aspect_sentiment_dict, **aspect_sentiment}
        aspect_sentiment_str = "["
        for aspect in aspect_sentiment.keys():
            aspect_sentiment_str += f'"{aspect}":"{aspect_sentiment[aspect]}";'
        aspect_sentiment_str = aspect_sentiment_str[:-1] + "]"
        #knowledge_sentence += doc_knowledge_sentence + "\n"
        knowledge_sentence += doc_knowledge_sentence + aspect_sentiment_str + "\n"
        knowledge_sentence = knowledge_sentence
    '''
    if do_summary == True:
        knowledge_sentence = summarizer(knowledge_sentence)[0]['summary_text'] + "\n"
    else:
        knowledge_sentence = knowledge_sentence
    '''
    return knowledge_sentence[:-1]


def create_knowledge_sentence(knowledge_list):
    #对knowledge_list进行排序，faq排在前面，按照entity_id,doc_id,sent_id排序
    knowledge_list = sorted(knowledge_list, key=lambda x: (x['doc_type'] != 'faq',x['entity_id'], x['doc_id'], x.get('sent_id', 0)))

    if len(knowledge_list) > sum_num:
        do_summary = True
    else:
        do_summary = False

    knowledge_sentence = ""
    entity_set = set()
    for knowledge in knowledge_list:
        entity_set.add(f"{knowledge['domain']}_{knowledge['entity_id']}")
    for entity in entity_set:
        entity_name = knowledge_with_absa[entity.split("_")[0]][str(entity.split("_")[1])]["name"]
        knowledge_sentence += f"{entity_name}\n"
        entity_knowledge_list = [knowledge_snippet for knowledge_snippet in knowledge_list if f"{knowledge_snippet['domain']}_{knowledge_snippet['entity_id']}" == entity]
        
        knowledge_sentence += create_knowledge_sentence_for_entity(entity_knowledge_list,do_summary) + "\n"
        #去掉最后一个换行符
    knowledge_sentence = knowledge_sentence[:-1]
    return knowledge_sentence
                


def create_few_shot(user_query):
    definition = "Definition:The output is a response to user query based on provided knowledge."
    #definition = "Definition: The output is a response to the user's query based on the provided knowledge. Note that after each review, I provide the aspects and sentiments of the review in the format ['aspect': 'sentiment'; 'aspect': 'sentiment']."
    '''
    query_embedding = similarity_model.encode([user_query])
    D, I = index.search(query_embedding, k=top_k_shot)
    '''
    random_indices = random.sample(range(len(resourse_labels_new)), top_k_shot)
    
    few_shot_text = ""
    for num,idx in enumerate(random_indices):
        resource_label = resourse_labels_new[idx]
        resource_log = resourse_logs_new[idx]
        query = resource_log[-1]["text"]
        knowledge = resource_label["knowledge"]
        response = resource_label["response"]
        knowledge_sentence = create_knowledge_sentence(knowledge)

        text = """Example {}-
input:
user query: {}
knowledge: 
{}
output:{}""".format(num+1,query,knowledge_sentence,response)
        few_shot_text += text + "\n"
    return definition + "\n" + few_shot_text


def create_input_text(user_query,knowledge):
    few_shot_text = create_few_shot(user_query)

    knowledge_sentence = create_knowledge_sentence(knowledge)

    user_query_text = """Now complete the following example-
input:
user query: {}
knowledge: 
{}
output:""".format(user_query,knowledge_sentence)
    input_text = few_shot_text + user_query_text
    #print(input_text)
    #以空格为单位切分
    text = input_text.split(" ")
    #print(len(input_text))
    #print(len(text))
    #print("--------------------------------------------------")
    return input_text



similarity_model = SentenceTransformer('/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/all-MiniLM-L6-v2')
#summarizer = pipeline("summarization", model="/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/MEETING_SUMMARY")
# huggingface hub model id

##加载资源向量库
resourse_vector = np.load('/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/resourse/resource_vector_base.npy')
dimension = resourse_vector.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(resourse_vector)

resourse_labels = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/resourse/labels.json')
resourse_logs = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/resourse/logs.json')
resourse_logs_aspect = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/resourse/logs_aspect.json')
knowledge_with_absa = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/knowledge_with_absa.json")

resourse_labels_new = []
resourse_logs_new = []
resourse_logs_aspect_new = []
for label,log,log_aspect in zip(resourse_labels,resourse_logs,resourse_logs_aspect):
    if label["target"] == True:
        resourse_labels_new.append(label)
        resourse_logs_new.append(log)
        resourse_logs_aspect_new.append(log_aspect)

#加载数据

log_file = f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/{dataset_type}/logs.json"
#label_file = f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/{dataset_type}/DebertaKS_QuaSiKS_reviewer_all_without_metadata.json"
'''
log_file = "/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/train/logs.json"
label_file = "/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/train/labels.json"
'''
label_file = f"./pred/dinstruct_gen/{dataset_type}/random_instruct_generation_llm.json"
logs = load_data(log_file)
labels = load_data(label_file)
'''
#为了便于区分，将原始数据中的response字段改为none
for label in tqdm(labels):
    if label["target"] == False:
        continue
    else:
        label["response"] ="none"
        #print(label["response"])
'''
count = 0  # 初始化计数器

for label,log in tqdm(zip(labels,logs),desc='process data'):
    if label["target"] == False:
        continue
    else:
        if label["response"] != "error":
            continue
        else:
            user_query = log[-1]["text"]
            knowledge = label["knowledge"]
            input_text = create_input_text(user_query,knowledge)
            #print(input_text)
            label["response"] = get_gpt_response(input_text)
        #print(label["response"])
    count += 1  # 更新计数器
    if count % 300 == 0:
        save_data(labels,f"./pred/dinstruct_gen/{dataset_type}/random_instruct_generation_llm.json")
        #print("input_text:",input_text)
        #print("response:",label["response"])
save_data(labels,f"./pred/dinstruct_gen/{dataset_type}/random_instruct_generation_llm.json")
