from datasets import Dataset, DatasetDict
import faiss
import numpy as np
import torch
import json
from tqdm import tqdm
import argparse
import random

from transformers import TrainerCallback, TrainerControl,EarlyStoppingCallback
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import BartForConditionalGeneration
from transformers import BartTokenizer
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--model_name', type=str, default='bart-base', help='Model name')
parser.add_argument('--withSelect', action='store_true', help='Select algorithm')
parser.add_argument('--withABSA', action='store_true', help='With absa')
parser.add_argument('--resultSavePath', type=str, default='output', help='Output path')
parser.add_argument('--device', type=str, default="cuda:1", help='Device')
args = parser.parse_args()

def loadDate(dataFile):
    dataList =[]
    for line in open(dataFile, "r"):
        dataList.append(json.loads(line))
    return dataList

tokenizer = BartTokenizer.from_pretrained(f'/home/zhuangjt/zhuangjt_disk3/SELECT/checkpoinnt/{args.model_name}')
model = BartForConditionalGeneration.from_pretrained(f'/home/zhuangjt/zhuangjt_disk3/SELECT/checkpoinnt/{args.model_name}').to(args.device)

if args.withSelect == True:
    similarity_model = SentenceTransformer('/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/all-MiniLM-L6-v2').to(args.device)
    resourse_vector = np.load('/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/resource_vector_base.npy')
    dimension = resourse_vector.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(resourse_vector)

    candidate_data = loadDate('/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/candidate_data.jsonl')

  
    def sort_by_similarity(text):
        query = similarity_model.encode([text])
        distances, indices = index.search(query, 1) 
        return indices[0][0]

    
def create_dataset(data):
    dataset = []
    for item in data:
        datadict = {}
        context = item['context']
        dialog = ""
        for i in range(len(context)):
            #按照i来区分user1和user2，并且换行
            if i % 2 == 0:
                dialog += "User1: " + context[i]['text'] + "\n"
            else:
                dialog += "User2: " + context[i]['text'] + "\n"
        #去掉最后一个换行符
        dialog = dialog[:-1]
        if args.withABSA == True:
            movie_user1 = item['userRoleMoviePerfenrnce']
            movie_user2 = item['systemRoleMoviePerfenrnce']
            
        if args.withSelect == True:
            #根据context的第一个句子，找到resource_vector中相似度最高的索引，然后找到对应的candidate_data
            similarDialog = ""
            index = sort_by_similarity(context[0]['text'])
            candidate = candidate_data[index]
            for i in range(len(candidate['context'])):
                if i % 2 == 0:
                    similarDialog += "User1: " + candidate['context'][i]['text'] + "\n"
                else:
                    similarDialog += "User2: " + candidate['context'][i]['text'] + "\n"
            similarDialog = similarDialog[:-1]
            similarReference = candidate['reference']['text']
            if args.withABSA == True:
                movie_similarUser1 = candidate['userRoleMoviePerfenrnce']
                movie_similarUser2 = candidate['systemRoleMoviePerfenrnce']
        
        input_text = ""
        defination = "Task definition: The output is a response to the user's query based on the user's preferences for movies."
        if args.withSelect == True:
            if args.withABSA == True:
                similarinput = """Example-
User1's movie preferences:
{movie_similarUser1}
User2's movie preferences:
{movie_similarUser2}
Dialogue:
{similarDialog}
Response:
{similarReference}
""".format(movie_similarUser1 = movie_similarUser1, movie_similarUser2 = movie_similarUser2, similarDialog = similarDialog, similarReference = similarReference)
                currentinput = """Now complete the following example-
User1's movie preferences:
{movie_user1}
User2's movie preferences:
{movie_user2}
Dialogue:
{dialog}
Response:
""".format(movie_user1 = movie_user1, movie_user2 = movie_user2, dialog = dialog)          
                input_text = defination + '\n' + similarinput + currentinput
            else:
                similarinput = """Example-
Dialogue:
{similarDialog}
Response:
{similarReference}
""".format(similarDialog = similarDialog, similarReference = similarReference)
                currentinput = """Now complete the following example-
Dialogue:
{dialog}
Response:
""".format(dialog = dialog)         
                input_text = defination + '\n' +similarinput + currentinput

        else:
            if args.withABSA == True:
                currentinput = """User1's movie preferences:
{movie_user1}
User2's movie preferences:
{movie_user2}
Dialogue:
{dialog}
Response:
""".format(movie_user1 = movie_user1, movie_user2 = movie_user2, dialog = dialog)          
                input_text = defination + '\n' + currentinput
            else:
                currentinput = """Dialogue:
{dialog}
Response:
""".format(dialog = dialog)   
                input_text = defination + '\n' + currentinput
            
        
        
        datadict["source"] = input_text
        datadict["target"] = item['reference']['text']
        

        dataset.append(datadict)
    return dataset


test_File = '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/val_data.jsonl'
test_data = loadDate(test_File)
test_dataset = create_dataset(test_data)

def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]
    
batch_size = 8

test_batches = list(batch(test_dataset, batch_size))

responses = []
for test_batch in tqdm(test_batches):
    for item in test_batch:
        inputs = tokenizer(item['source'], return_tensors="pt", max_length=1024, truncation=True, padding="max_length").to(args.device)
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, num_beams=4, max_length=100, early_stopping=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        responses.append(response)

with open(f'/home/zhuangjt/zhuangjt_disk3/SELECT/result/val/{args.resultSavePath}.jsonl', 'w') as f:
    for response in responses:
        f.write(json.dumps({"response": response}) + '\n')

