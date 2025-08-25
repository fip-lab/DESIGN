from datasets import Dataset, DatasetDict
import numpy as np
import json
from tqdm import tqdm
import argparse

from rouge_score import rouge_scorer

from summ_eval.bleu_metric import BleuMetric
from summ_eval.meteor_metric import MeteorMetric


parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--label_file', type=str, help='Label file name')
parser.add_argument('--hyp_file', type=str, help='Hyp file name')
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


def cal_rouge(ref_response, hyp_response):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = scorer.score(ref_response, hyp_response)

    
    rougeL = scores['rougeL'].fmeasure

    return rougeL

def cal_bleu(ref_response, hyp_response):
    bleu_metric = BleuMetric()
    bleu_score = bleu_metric.evaluate_example(hyp_response, ref_response)['bleu']
    return bleu_score

def cal_meteor(ref_response, hyp_response):
    meteor_metric = MeteorMetric()
    meteor_score = meteor_metric.evaluate_example(hyp_response,ref_response)['meteor']
    return meteor_score

    


# 批处理函数
def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


'''
label_file = args.label_file
hyp_file = args.hyp_file
out_file = args.out_file
'''

label_file = "/home/zhuangjt/zhuangjt_disk3/SK-TOD/DSTC10_data/test/labels.json"
hyp_file="/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/pred/DSTC10/test/bart_large_withoutabsa_1shot.json"
out_file="/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/DynamicBart/pred/DSTC10/test/bart_large_withoutabsa_1shot_bmr_score.json"


batch_size = 8
labels = load_data(label_file)
hyp = load_data(hyp_file)

ref_responses = []
pred_responses = []
count = 0

for label, hyp in tqdm(zip(labels, hyp)):
    if label["target"] is True:
        ref_responses.append(label["response"])
        pred_responses.append(hyp["response"])
        count += 1

            

#保存平均值到out_file

with open(out_file, 'w') as file:
    json.dump({"rouge":rouge_scores_mean, "bleu":bleu_scores_mean, "meteor":meteor_scores_mean}, file, indent=4)
 