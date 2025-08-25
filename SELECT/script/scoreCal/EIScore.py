import fed
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
import json
import openai
import logging
import time

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--response_file', type=str, help='Label file name')
parser.add_argument('--dialogue_file', type=str, help='Input file name')
parser.add_argument('--out_file', type=str, help='Output file name')
parser.add_argument('--device', type=str, default="cuda:0", help='Device')
args = parser.parse_args()



entailment_model_name = "/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
entailment_model = AutoModelForSequenceClassification.from_pretrained(entailment_model_name).to(args.device)
entailment_tokenizer = AutoTokenizer.from_pretrained(entailment_model_name)

interest_model, interest_tokenizer = fed.load_models(args.device, "/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/DialoGPT-large")


def loadDate(dataFile):
    dataList =[]
    for line in open(dataFile, "r"):
        dataList.append(json.loads(line))
    return dataList

def calculate_entailment(knowledge, response):
    input = entailment_tokenizer(knowledge, response, truncation=True, return_tensors="pt").to(args.device)
    output = entailment_model(input["input_ids"].to(args.device))  # device = "cuda:0" or "cpu"
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    return prediction['entailment']

def calculate_interest(response):
    input = "<|endoftext|>" + response
    score = fed.evaluate(input,
                      interest_model,
                      interest_tokenizer,
                      args.device)["interesting"]
    normalized_score = ((score + 1) / 2) * 100
    return normalized_score



dialogue_data = loadDate(args.dialogue_file)
response_data = loadDate(args.response_file)

entailment_scores_list = []
interest_scores_list = []
for i in range(len(dialogue_data)):
    knowledge = """User1's movie preferences:
{movie_similarUser1}
User2's movie preferences:
{movie_similarUser2}""".format(movie_similarUser1 = dialogue_data[i]['userRoleMoviePerfenrnce'], movie_similarUser2 = dialogue_data[i]['systemRoleMoviePerfenrnce'])
    response = response_data[i]['response']
    entailment_score = calculate_entailment(knowledge, response)
    interest_score = calculate_interest(response)
    entailment_scores_list.append(entailment_score)
    interest_scores_list.append(interest_score)


entailment_score = sum(entailment_scores_list) / len(entailment_scores_list)
interest_score = sum(interest_scores_list) / len(interest_scores_list)

with open(args.out_file, "w") as f:
    f.write(json.dumps({"entailment": entailment_score, "interest": interest_score}))
    f.write("\n")

