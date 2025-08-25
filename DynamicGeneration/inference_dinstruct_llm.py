#import openai
from openai import OpenAI
import os
import json
import pandas as pd
import numpy as np
import logging

import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
import argparse

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--model_name', type=str, help='Model name')
parser.add_argument('--base_url', type=str, help='Base url')
parser.add_argument('--api_key', type=str, help='Api key')
parser.add_argument('--save_file', type=str, help='Save file')
args = parser.parse_args()

def get_response(prompt):
    try:
        client = OpenAI(
            base_url=args.base_url,
            api_key=args.api_key
        )
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model=args.model_name,
            messages=messages
        )
        result = completion.choices[0].message.content
    except Exception as e:
        logging.info(e)
        result = "error"
        time.sleep(60)
    
    return result



dataset_type = "test"
top_k_shot = 1

def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def save_data(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def create_knowledge_sentence_for_entity(entity_knowledge_list):
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

    return knowledge_sentence[:-1]


def create_knowledge_sentence(knowledge_list):
    #对knowledge_list进行排序，faq排在前面，按照entity_id,doc_id,sent_id排序
    knowledge_list = sorted(knowledge_list, key=lambda x: (x['doc_type'] != 'faq',x['entity_id'], x['doc_id'], x.get('sent_id', 0)))

    knowledge_sentence = ""
    entity_set = set()
    for knowledge in knowledge_list:
        entity_set.add(f"{knowledge['domain']}_{knowledge['entity_id']}")
    for entity in entity_set:
        entity_name = knowledge_with_absa[entity.split("_")[0]][str(entity.split("_")[1])]["name"]
        knowledge_sentence += f"{entity_name}\n"
        entity_knowledge_list = [knowledge_snippet for knowledge_snippet in knowledge_list if f"{knowledge_snippet['domain']}_{knowledge_snippet['entity_id']}" == entity]
        
        knowledge_sentence += create_knowledge_sentence_for_entity(entity_knowledge_list) + "\n"
        #去掉最后一个换行符
    knowledge_sentence = knowledge_sentence[:-1]
    return knowledge_sentence
                


def create_few_shot(user_query):
    #definition = "Definition:The output is a response to user query based on provided knowledge."
    definition = "Definition: The output is a response to the user's query based on the provided knowledge. Note that after each review, I provide the aspects and sentiments of the review in the format ['aspect': 'sentiment'; 'aspect': 'sentiment']."
    query_embedding = similarity_model.encode([user_query])
    D, I = index.search(query_embedding, k=top_k_shot)
    
    few_shot_text = ""
    for num,idx in enumerate(I[0]):
        resource_label = resourse_labels_new[idx]
        resource_log = resourse_logs_new[idx]
        #query = resource_log[-1]["text"]
        #dialog = resource_log[:-4]
        #取倒数4句
        dialog = resource_log[-3:]
        #根据顺序，分别写成
        #[user] xxx
        #[agent] xxx
        #[user] xxx
        #[agent] xxx
        query = ""
        for turn in dialog:
            if turn["speaker"] == "U":
                query += "[user] " + turn["text"] + "\n"
            else:
                query += "[agent] " + turn["text"] + "\n"
        knowledge = resource_label["knowledge"]
        response = resource_label["response"]
        knowledge_sentence = create_knowledge_sentence(knowledge)

        text = """Example {}-
input:
dialog: 
{}
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
dialog: {}
knowledge: 
{}
output:""".format(user_query,knowledge_sentence)
    input_text = few_shot_text + user_query_text
    text = input_text.split(" ")

    return input_text



similarity_model = SentenceTransformer('/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/all-MiniLM-L6-v2')

##加载资源向量库
resourse_vector = np.load('/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/resourse/resource_vector_base.npy')
dimension = resourse_vector.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(resourse_vector)

resourse_labels = load_data('/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/resourse/labels.json')
resourse_logs = load_data('/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/resourse/logs.json')

knowledge_with_absa = load_data("/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/data/knowledge_with_absa.json")

resourse_labels_new = []
resourse_logs_new = []
for label,log in zip(resourse_labels,resourse_logs):
    if label["target"] == True:
        resourse_labels_new.append(label)
        resourse_logs_new.append(log)

#加载数据

log_file = f"/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/data/{dataset_type}/logs.json"
label_file = f"/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/data/{dataset_type}/labels.json"

logs = load_data(log_file)
labels = load_data(label_file)

count = 0  # 初始化计数器

for label,log in tqdm(zip(labels,logs),desc='process data'):
    if label["target"] == False:
        continue
    else:
        user_query = log[-1]["text"]
        knowledge = label["knowledge"]
        input_text = create_input_text(user_query,knowledge)
        label["response"] = get_response(input_text)
    count += 1  # 更新计数器
    if count % 150 == 0:
        save_data(labels,f"./pred/dinstruct_gen/{dataset_type}/{args.save_file}")
save_data(labels,f"./pred/dinstruct_gen/{dataset_type}/{args.save_file}")
