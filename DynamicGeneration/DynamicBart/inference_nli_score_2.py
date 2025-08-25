from datasets import Dataset, DatasetDict
import faiss
import numpy as np
import torch
import json
from tqdm import tqdm
import argparse
import random

import fed

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSequenceClassification, AutoTokenizer
from transformers import BartForConditionalGeneration
from transformers import BartTokenizer
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--label_ks', type=str, help='Label file name')
parser.add_argument('--out_file', type=str, help='Output file name')
parser.add_argument('--device', type=str, default="cuda:0", help='Device')
args = parser.parse_args()




def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def save_data(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


device = args.device


entailment_model_name = "/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
entailment_model = AutoModelForSequenceClassification.from_pretrained(entailment_model_name).to(device)
entailment_tokenizer = AutoTokenizer.from_pretrained(entailment_model_name)

interest_model, interest_tokenizer = fed.load_models(device, "/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/DialoGPT-large")


knowledge_with_absa = load_data("/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/data/knowledge_with_absa.json")
knowledge_with_absa_file = load_data("/disk4/zhuangjt/zhuangjt_disk3/SK-TOD/data/knowledge_with_absa.json")






def calculate_entailment(knowledge, response):
    input = entailment_tokenizer(knowledge, response, truncation=True, return_tensors="pt").to(device)
    output = entailment_model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    
    return prediction['entailment']

def calculate_interest(response):
    input = "<|endoftext|>" + response
    score = fed.evaluate(input,
                      interest_model,
                      interest_tokenizer,
                      device)["interesting"]
    normalized_score = ((score + 1) / 2) * 100
    return normalized_score


    
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



label_file = args.label_ks
out_file = args.out_file

batch_size = 8
labels = load_data(label_file)
labels_batches = list(batch(labels, batch_size))


entailment_scores_list = []
interest_scores_list = []

for batch_idx, labels_batch in enumerate(labels_batches):
    for label_idx, label in enumerate(tqdm(labels_batch, desc=f'Processing batch {batch_idx}')):
        if label["target"] == False:
            continue
        else:
            knowledge_text = create_knowledge_sentence(label["knowledge"])
            response = label["response"]
            entailment_score = calculate_entailment(knowledge_text, response)
            interest_score = calculate_interest(response)

            #entailment_scores = [calculate_entailment(knowledge_text, response) for response in responses]
            #interest_scores = [calculate_interest(response) for response in responses]
            entailment_scores_list.append(entailment_score)
            interest_scores_list.append(interest_score)

#计算entailment和interest的平均值
entailment_scores_list = np.array(entailment_scores_list)
interest_scores_list = np.array(interest_scores_list)
entailment_scores_mean = np.mean(entailment_scores_list, axis=0)
interest_scores_mean = np.mean(interest_scores_list, axis=0)
            
#保存平均值到out_file

with open(out_file, 'w') as file:
    json.dump({"entailment":entailment_scores_mean.tolist(), "interest":interest_scores_mean.tolist()}, file, indent=4)
 