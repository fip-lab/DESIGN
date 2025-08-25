import logging
import argparse
from bert_score import score as bertScore
from pycocotools.coco import COCO
from pycocoevalcap.cider.cider import Cider
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json


parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--hyp_file', type=str, help='Label file name')
parser.add_argument('--ref_file', type=str, help='Label file name')
parser.add_argument('--out_file', type=str, help='Output file name')
args = parser.parse_args()



#计算函数
def calBertScore(hyp, ref):
    P, R, F1 = bertScore(hyp, ref, lang="en", verbose=True)
    return F1.mean()



def calCiderScore(hyp, ref):
    coco = COCO()
    cider_scorer = Cider()
    scores, _ = cider_scorer.compute_score(hyp, ref)
    return scores

def calGleuScore(hyp, ref):
    smoothing_function = SmoothingFunction().method1
    gleu_scores = []
    for ref_list, hyp in zip(ref, hyp):
        score = sentence_bleu(ref_list, hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing_function)
        gleu_scores.append(score)
    average_gleu = sum(gleu_scores) / len(gleu_scores)
    return average_gleu


##构建数据
def loadDate(dataFile):
    dataList =[]
    for line in open(dataFile, "r"):
        dataList.append(json.loads(line))
    return dataList


hyp_data = loadDate(args.hyp_file)
ref_data = loadDate(args.ref_file)


'''
for bert_score and bleurt
hypotheses = ["the cat is on the mat", "there is a cat on the mat"]
references = ["there is a cat on the mat", "the cat is on the mat"]
'''
hyp4bert = []
ref4bert = []

'''
for cider
annotations = {
    'image1': [ 'A dog running with a ball'],
    'image2': [ 'A cat resting on the couch']
}
predictions = {
    'image1': ['A dog playing with a ball'],
    'image2': ['A cat sleeping on the sofa']
}
'''
hyp4cider = {}
ref4cider = {}


'''
#for gleu
references = [
    ["there was a cat on the mat"],
    ["he read a book"]
]
hypotheses = [
    "the cat was sitting on the mat",
    "he read a book"
]
'''
hyp4gleu = []
ref4gleu = []

ciderId = 0
for i in range(len(hyp_data)):
    ref = ref_data[i]['reference']['text']
    hyp = hyp_data[i]['response']
    hyp4bert.append(hyp)
    ref4bert.append(ref)
    hyp4cider[str(ciderId)] = [hyp]
    ref4cider[str(ciderId)] = [ref]
    hyp4gleu.append(hyp)
    ref4gleu.append([ref])
    ciderId += 1


bert_Score = calBertScore(hyp4bert, ref4bert)
cider_Score = calCiderScore(hyp4cider, ref4cider)
gleu_Score = calGleuScore(hyp4gleu, ref4gleu)

result = {
    "bertScore": bert_Score.tolist(),
    "ciderScore": cider_Score,
    "gleuScore": gleu_Score
}


with open(args.out_file, 'w') as file:
    json.dump(result, file)

