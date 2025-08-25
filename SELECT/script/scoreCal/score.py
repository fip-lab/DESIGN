import json
from rouge_score import rouge_scorer
import summ_eval
from summ_eval.bleu_metric import BleuMetric
from summ_eval.meteor_metric import MeteorMetric
import argparse


parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--resultSavePath', type=str, default='output', help='Output path')
args = parser.parse_args()

def loadDate(dataFile):
    dataList =[]
    for line in open(dataFile, "r"):
        dataList.append(json.loads(line))
    return dataList

test_File = '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/val_data.jsonl'
result_File = '/home/zhuangjt/zhuangjt_disk3/SELECT/result/val/' + args.resultSavePath + '.jsonl'
test_data = loadDate(test_File)
result = loadDate(result_File)


rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bleu_metric = BleuMetric()
meteor_metric = MeteorMetric()

rouge1 = 0
rouge2 = 0
rougeL = 0
bleu = 0
meteor = 0

for i in range(len(test_data)):
    ref = test_data[i]['reference']['text']
    hyp = result[i]['response']
    rouge = rouge_scorer.score(ref, hyp)
    rouge1 += rouge['rouge1'].fmeasure
    rouge2 += rouge['rouge2'].fmeasure
    rougeL += rouge['rougeL'].fmeasure
    bleu += bleu_metric.evaluate_example(hyp,ref)['bleu']
    meteor += meteor_metric.evaluate_example(hyp,ref)['meteor']

rouge1 = rouge1 / len(test_data)
rouge2 = rouge2 / len(test_data)
rougeL = rougeL / len(test_data)

bleu = bleu / len(test_data)
meteor = meteor / len(test_data)

result = {}
result['rouge1'] = rouge1
result['rouge2'] = rouge2
result['rougeL'] = rougeL
result['bleu'] = bleu
result['meteor'] = meteor

savePath = '/home/zhuangjt/zhuangjt_disk3/SELECT/score/val' + args.resultSavePath + '.json'
with open(savePath, 'w') as f:
    json.dump(result, f, indent=4)
