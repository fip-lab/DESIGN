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


entailment_model_name = "/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
entailment_model = AutoModelForSequenceClassification.from_pretrained(entailment_model_name).to(device)
entailment_tokenizer = AutoTokenizer.from_pretrained(entailment_model_name)

interest_model, interest_tokenizer = fed.load_models(device, "/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/DialoGPT-large")


knowledge_base= load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/DSTC10_data/knowledge.json")




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


def create_knowledge_sentence(knowledge_list):
    knowledge_sentence = ""
    for knowledge in knowledge_list:
        question = knowledge_base[knowledge["domain"]][str(knowledge["entity_id"])]["docs"][str(knowledge["doc_id"])]["title"]
        answer = knowledge_base[knowledge["domain"]][str(knowledge["entity_id"])]["docs"][str(knowledge["doc_id"])]["body"]
        knowledge_sentence += "Q: " + question + "\n" + "A: " + answer + "\n"
    return knowledge_sentence[:-1]

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
 