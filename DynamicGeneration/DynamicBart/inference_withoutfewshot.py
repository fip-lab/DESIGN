from datasets import Dataset, DatasetDict
import faiss
import numpy as np
import torch
import json
from tqdm import tqdm
import argparse
import random

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import BartForConditionalGeneration
from transformers import BartTokenizer
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--top_k_shot', type=int, default=3, help='Top k shot')
parser.add_argument('--model_path', type=str, default='bart-base', help='Model name')
parser.add_argument('--dataset', type=str, default="test", help='Dataset type')
parser.add_argument('--label_ks', type=str, help='Label file name')
parser.add_argument('--out_file', type=str, help='Output file name')
parser.add_argument('--device', type=str, default="cuda:1", help='Device')
parser.add_argument('--withabsa', type=bool, default=False, help='Whether to use absa')
parser.add_argument('--select_algorithm', type=str, default="SELECT", help='Whether to use absa')
args = parser.parse_args()


def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def save_data(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

dataset_type=args.dataset
top_k_shot = args.top_k_shot
model_path = args.model_path  # 修改为你的训练输出目录
device = args.device
withabsa = args.withabsa
model = BartForConditionalGeneration.from_pretrained(model_path).to(device)
tokenizer = BartTokenizer.from_pretrained(model_path)
similarity_model = SentenceTransformer('/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/all-MiniLM-L6-v2').to(device)
##加载资源向量库
resourse_vector = np.load('/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/Dynamic_Dataset/resourse/resource_vector_base.npy')
dimension = resourse_vector.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(resourse_vector)

resourse_labels = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/Dynamic_Dataset/resourse/labels.json')
resourse_logs = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/Dynamic_Dataset/resourse/logs.json')
resourse_logs_aspect = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/Dynamic_Dataset/resourse/logs_aspect.json')
resourse_labels_new = []
resourse_logs_new = []
resourse_logs_aspect_new = []
for label,log,log_aspect in zip(resourse_labels,resourse_logs,resourse_logs_aspect):
    if label["target"] == True:
        resourse_labels_new.append(label)
        resourse_logs_new.append(log)
        resourse_logs_aspect_new.append(log_aspect)

knowledge_with_absa = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/knowledge_with_absa.json")
knowledge_with_absa_file = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/knowledge_with_absa.json")


def create_few_shot(user_query):
    if withabsa:
        #definition = "Definition:The output is a response to user query based on provided knowledge. Note that after each review, I provide the aspects and sentiments of the review in the format ['aspect': 'sentiment'; 'aspect': 'sentiment']."
        definition = "Definition: The output is a response to the user's query based on the provided knowledge. Note that after each review, I provide the aspects and sentiments of the review in the format ['aspect': 'sentiment'; 'aspect': 'sentiment']."
    else:
        definition = "Definition: The output is a response to the user's query based on the provided knowledge."
    query_embedding = similarity_model.encode([user_query])
    D, I = index.search(query_embedding, k=top_k_shot)
    
    few_shot_text = ""
    if args.select_algorithm == 'SELECT':
        query_embedding = similarity_model.encode([user_query])
        D, I = index.search(query_embedding, k=top_k_shot)
        for num,idx in enumerate(I[0]):
            resource_label = resourse_labels_new[idx]
            resource_log = resourse_logs_new[idx]
            query = resource_log[-1]["text"]
            knowledge = resource_label["knowledge"]
            response = resource_label["response"]
            knowledge_sentence = create_knowledge_sentence(knowledge)

            text = """Example {}-
Knowledge: 
{}
Query:{}
Answer:{}""".format(num+1,knowledge_sentence,query,response)
            few_shot_text += text + "\n"
    
    elif args.select_algorithm == 'RANDOM':
        for num in range(top_k_shot):
            idx = random.randint(0,len(resourse_labels_new)-1)
            resource_label = resourse_labels_new[idx]
            resource_log = resourse_logs_new[idx]
            query = resource_log[-1]["text"]
            knowledge = resource_label["knowledge"]
            response = resource_label["response"]
            knowledge_sentence = create_knowledge_sentence(knowledge)

            text = """Example {}-
Knowledge:
{}
Query:{}
Answer:{}""".format(num+1,knowledge_sentence,query,response)
        few_shot_text += text + "\n"

    return definition + "\n" + few_shot_text
    #return few_shot_text
    
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
        aspect_sentiment_str = "["
        for aspect in aspect_sentiment.keys():
            aspect_sentiment_str += f'"{aspect}":"{aspect_sentiment[aspect]}";'
        aspect_sentiment_str = aspect_sentiment_str[:-1] + "]"
        if withabsa:
            knowledge_sentence += doc_knowledge_sentence + aspect_sentiment_str + "\n"
        else:
            knowledge_sentence += doc_knowledge_sentence + "\n"
    
        knowledge_sentence = knowledge_sentence
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

# 批处理函数
def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]

    
log_file = f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/{dataset_type}/logs.json"
label_file = args.label_ks
out_file = args.out_file

batch_size = 8
labels = load_data(label_file)
logs = load_data(log_file)
labels_batches = list(batch(labels, batch_size))
logs_batches = list(batch(logs, batch_size))

for batch_idx, (labels_batch, logs_batch) in enumerate(zip(labels_batches, logs_batches)):
    for label_idx, (label, log) in enumerate(tqdm(zip(labels_batch, logs_batch), desc=f'Processing batch {batch_idx}')):
        if label["target"] == False:
            continue
        else:
            query = log[-1]["text"]
            few_shot_text = create_few_shot(query)
            knowledge_text = create_knowledge_sentence(label["knowledge"])
            user_query_text = """Complete the following dialogue-
Knowledge: 
{}
Query:{}
Answer:""".format(knowledge_text,query)
            input_text = user_query_text
            inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding="max_length").to(device)
            with torch.no_grad():
                outputs = model.generate(inputs.input_ids, num_beams=4, max_length=100, early_stopping=True)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            label["response"] = response
            
            #print(query)
            #print(response)
            #print("---------------------------------------------------")

save_data(labels, out_file)