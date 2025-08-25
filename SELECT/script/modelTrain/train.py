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
parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--model_name', type=str, default='bart-base', help='Model name')
parser.add_argument('--withSelect', action='store_true', help='Select algorithm')
parser.add_argument('--withABSA', action='store_true', help='With absa')
parser.add_argument('--modelSavePath', type=str, default='output', help='Output path')
args = parser.parse_args()

#希望withSelect和withABSA默认为False，则不需要在命令行传递它们。
#希望它们为True，则只需在命令行中使用--withSelect和--withABSA即可，而不需要显式指定True或False


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def loadDate(dataFile):
    dataList =[]
    for line in open(dataFile, "r"):
        dataList.append(json.loads(line))
    return dataList


tokenizer = BartTokenizer.from_pretrained(f'/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/{args.model_name}')
model = BartForConditionalGeneration.from_pretrained(f'/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/{args.model_name}')

if args.withSelect == True:
    similarity_model = SentenceTransformer('/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/all-MiniLM-L6-v2')
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
            
        
        print(input_text)
        datadict["source"] = input_text
        datadict["target"] = item['reference']['text']
        

        dataset.append(datadict)
    return dataset





def list_to_dict(list_data):
    # 初始化一个字典，键为'source'和'target'，值为空列表
    dict_data = {"source": [], "target": []}
    # 遍历列表，分别将'source'和'target'的值追加到相应的列表中
    for item in list_data:
        dict_data["source"].append(item["source"])
        dict_data["target"].append(item["target"])
    return dict_data


# 将列表转换为字典
train_File = '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/train_data.jsonl'
val_File = '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/val_data.jsonl'
train_data = loadDate(train_File)
val_data = loadDate(val_File)
train_dataset = create_dataset(train_data)
val_dataset = create_dataset(val_data)
train_data_dict = list_to_dict(train_dataset)
val_data_dict = list_to_dict(val_dataset)


trainDataset = Dataset.from_dict(train_data_dict)
valDataset = Dataset.from_dict(val_data_dict)


datasets = DatasetDict({
'train': trainDataset,
'validation': valDataset,
})


def preprocess_function(examples):
    inputs = examples["source"]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")

    # 我们需要将targets也编码（包括前缀空间）
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=1024, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = datasets.map(preprocess_function, batched=True)


training_args = Seq2SeqTrainingArguments(
    output_dir= args.modelSavePath,                 # 输出目录
    fp16=True,                              # 是否使用混合精度训练
    per_device_train_batch_size=2,          # 每个GPU的训练批次大小
    per_device_eval_batch_size=4,           # 每个GPU的评估批次大小
    gradient_accumulation_steps=4,          # 梯度累积步骤
    learning_rate=3e-5,                     # 学习率    learning_rate=3e-5,                     # 学习率
    
    adam_epsilon=1e-8,                      # Adam优化器的epsilon值
    max_grad_norm=1.0,                      # 梯度裁剪的最大梯度范数
    num_train_epochs=100,                    # 训练轮数
    warmup_ratio=0.2,                       # 预热比例，如果使用了warmup_ratio，则无需设置warmup_steps
    evaluation_strategy="epoch",            # 评估策略，可以是"steps"或"epoch"
    save_strategy="epoch",                  # 保存策略，与评估策略相同
    save_total_limit=2,                     # 保存的最大模型数量
    load_best_model_at_end=True,            # 训练结束时加载最佳模型
    metric_for_best_model="loss",           # 选择最佳模型的指标，默认是loss，也可以根据需要设置其他指标
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()


 
    

