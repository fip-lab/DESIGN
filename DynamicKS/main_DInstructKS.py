import json
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import faiss
import argparse
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)

from scripts.knowledge_reader import KnowledgeReader

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 如果你的两张卡的编号是0和1

parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--positive_sample_size', type=int, default=3, help='Number of positive samples')
args = parser.parse_args()
positive_sample_size = args.positive_sample_size

#negative_sample_size = 2

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

knowledge_reader = KnowledgeReader("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data","knowledge_with_absa.json")

def prepare_knowledge():
    knowledge_docs = get_snippet_list()
    knoweldege_snippets = {}
    for snippet_id, snippet in tqdm(enumerate(knowledge_docs), disable=False, desc='prepare knowledge snippets '):
            key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
            knoweldege_snippets[key] = snippet["doc"]["body"]
    return  knoweldege_snippets

def get_snippet_list():
    result = []
    for domain in knowledge_reader.get_domain_list():
        for entity_id in knowledge_reader.knowledge[domain].keys():
            for review_doc_id in knowledge_reader.get_review_doc_ids(domain, entity_id):
                review_doc = knowledge_reader.get_review_doc(domain, entity_id, review_doc_id)
                for review_sent_id, review_sent in review_doc['sentences'].items():
                    review_absa = knowledge_reader.get_review_absa(domain, entity_id, review_doc_id, review_sent_id)
                    result.append(
                        {'domain': domain, 'entity_id': entity_id, 'entity_name': review_doc['entity_name'],
                            'doc_id': f"{review_doc_id}-{review_sent_id}",
                            'doc': {'body': review_sent},
                            'absa': review_absa})
            for faq_doc_id in knowledge_reader.get_faq_doc_ids(domain, entity_id):
                faq_doc = knowledge_reader.get_faq_doc(domain, entity_id, faq_doc_id)
                faq_absa = knowledge_reader.get_faq_absa(domain, entity_id, faq_doc_id)
                #print("faq_absa:",faq_absa)
                result.append({'domain': domain, 'entity_id': entity_id, 'entity_name': faq_doc['entity_name'],
                                'doc_id': faq_doc_id,
                                'doc': {'body': f"{faq_doc['question']} {faq_doc['answer']}"},
                                'absa': faq_absa})
    return result


knowledge_snippets = prepare_knowledge() #key是domain__entity_id__doc_id,value是knowledge sentence

##few-shot构建函数
def create_few_shot(user_query, aspect_item_of_query, knowledge_sentence):
    ##这个函数主要是用于构建few-shot
    ##输入是用户的query
    ##然后根据用户的query，从resourse中找到最相似的query向量的索引
    ##并根据索引找到对应的query文本和aspect文本，knowledge标签字典
    ##从knowledge标签字典中随机选取两个作为positive,再到对应实体的知识列表中随机选择两个不在knowledge标签字典中的作为negative
    ##最后返回一个few-shot的列表，包含7个元素，分别是Definition,positive1,positive2,negative1,negative2,user_query,end_instruction
    definition = "Definition: The output will be an assessment of whether the knowledge can be used to support answering the user's query.If it can support, the output will be 'yes'; otherwise, the output will be 'no'."
    user_query_text = """Now complete the following example-
input:
[user_query]{}
[aspect_item_of_query]{}
[knowledge]{}
output:""".format(user_query,aspect_item_of_query,knowledge_sentence)
    
    ##计算query的向量
    query_embedding = similarity_model.encode([user_query])
    D, I = index.search(query_embedding, k=1)
    ##索引是I[0][0]
    resourse_labels = resourse_labels_new[I[0][0]]
    resourse_logs = resourse_logs_new[I[0][0]][-1]["text"]
    resourse_logs_aspect = resourse_logs_aspect_new[I[0][0]]

    
    #构建knowledge snippets（在最开始构建）,key是domain__entity_id__doc_id,value是knowledge sentence
    #在每一轮构建中，将正样本按照domain__entity_id__doc_id构建，获取相对应的knowledge sentence
    #同时，也要有一个正样本key_list,用于之后的随机抽取负样本的排除工作
    #获取knowledge_prefix_visited中的所有元素，然后从knowledge snippets中找到前缀是这些元素的key
    #然后将正样本的key_list中的元素从knowledge snippets中删除
    #然后从剩下的key中随机抽取两个作为负样本
    
    sample_knowledges = resourse_labels["knowledge"]
    positive_knowledges = []
    for knowledge in sample_knowledges:
        if knowledge["doc_type"] == "review":
            key = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}-{knowledge['sent_id']}"
        else:
            key = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}"
        positive_knowledges.append(key)

    knowledge_prefix_visited = set()
    knowledge_candidates = []
    for knowledge in sample_knowledges:
        prefix = "{}__{}".format(knowledge["domain"], knowledge["entity_id"])
        if prefix not in knowledge_prefix_visited:
            knowledge_prefix_visited.add(prefix)
            candidates = [
                cand 
                for cand in knowledge_snippets.keys()
                if "__".join(cand.split("__")[:-1]) == prefix
            ]
            knowledge_candidates.extend(candidates)
    
    #negative_knowledges是从knowledge_candidates中删除掉positive_knowledges的key_list中的元素之后的剩余元素中随机抽取两个
    negative_knowledges_text = []
    for knowledge in knowledge_candidates:
        if knowledge not in positive_knowledges:
            negative_knowledges_text.append(knowledge_snippets[knowledge])
    ##获取对应的knowledge sentence
    positive_knowledges_text = []
    for knowledge in positive_knowledges:
        positive_knowledges_text.append(knowledge_snippets[knowledge])
    if len(positive_knowledges_text) >=positive_sample_size:
        positive_knowledges_text = random.sample(positive_knowledges_text,positive_sample_size)
    else:
        positive_knowledges_text = positive_knowledges_text
    '''
    if len(negative_knowledges_text) >=negative_sample_size:
        negative_knowledges_text = random.sample(negative_knowledges_text,negative_sample_size)
    else:
        negative_knowledges_text = negative_knowledges_text
    '''
    ##前面已经定义好Definition和Now complete the following example-之后的文本，所以只需要完成中间两个positive和negative的文本
    positive_text = ""
    for i in range(len(positive_knowledges_text)):
        positive_text += "Positive example {}-\ninput:\n[user_query]{}\n[aspect_item_of_query]{}\n[knowledge]{}\noutput:yes\n".format(i+1,resourse_logs,resourse_logs_aspect,positive_knowledges_text[i])
    '''
    negative_text = ""
    for i in range(len(negative_knowledges_text)):
        negative_text += "Negative example {}-\ninput:\n[user_query]{}\n[aspect_item_of_query]{}\n[knowledge]{}\noutput:no\n".format(i+1,resourse_logs,resourse_logs_aspect,negative_knowledges_text[i])

    input_text = definition + "\n" + positive_text + negative_text + user_query_text 
    '''
    input_text = definition + "\n" + positive_text  + user_query_text
    #print(input_text)

    return input_text

def process_data(data):
    inputs = []
    labels = []
    for item in tqdm(data,desc="process data"):
        inputs.append(create_few_shot(item['user_query'], item['aspect_item_of_query'], item['knowledge_sentence']))
        labels.append(item['labels'])
    return inputs, labels

# 1. 加载数据
def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def compute_metrics(p):
    pred_labels = [tokenizer.decode(label_id, skip_special_tokens=True).strip() for label_id in p.predictions.argmax(-1)]
    # 处理非预期输出
    pred_labels = [label if label in ["yes", "no"] else "no" for label in pred_labels]
    true_labels = p.label_ids
    true_labels = [tokenizer.decode(label_id, skip_special_tokens=True).strip() for label_id in true_labels]
    # 计算 TP, FP
    TP = sum([1 for pred, true in zip(pred_labels, true_labels) if pred == "yes" and true == "yes"])
    FP = sum([1 for pred, true in zip(pred_labels, true_labels) if pred == "yes" and true == "no"])
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    return {"precision_for_yes": precision}


class T5ClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = tokenizer(self.labels[idx], padding="max_length", truncation=True, max_length=512, return_tensors="pt").input_ids[0].clone().detach()
        #item['labels'] = torch.tensor(tokenizer(self.labels[idx], padding="max_length", truncation=True, max_length=512, return_tensors="pt").input_ids[0])
        return item

    def __len__(self):
        return len(self.labels)
    

similarity_model = SentenceTransformer('/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/all-MiniLM-L6-v2')
##加载资源向量库
resourse_vector = np.load('/home/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/resourse/resource_vector_base.npy')
dimension = resourse_vector.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(resourse_vector)

resourse_labels = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/resourse/labels.json')
resourse_logs = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/resourse/logs.json')
resourse_logs_aspect = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/resourse/logs_aspect.json')

##先对resourse进行处理，构建三个列表，把所有label["target"] == false的排除掉
resourse_labels_new = []
resourse_logs_new = []
resourse_logs_aspect_new = []
for label,log,log_aspect in zip(resourse_labels,resourse_logs,resourse_logs_aspect):
    if label["target"] == True:
        resourse_labels_new.append(label)
        resourse_logs_new.append(log)
        resourse_logs_aspect_new.append(log_aspect)



train_data = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/train_dataset_2.json")
val_data = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/val_dataset.json")

##改进:对训练集进行采样，选取所有labels为yes的数据，然后随机采样与正样本数量相同的负样本(labels为no)


train_texts, train_labels = process_data(train_data)
val_texts, val_labels = process_data(val_data)

# 3. 使用 T5 的 tokenizer 进行 tokenization
tokenizer = AutoTokenizer.from_pretrained('/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/flan-t5-base')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

# 4. 创建 PyTorch 数据集
train_dataset = T5ClassificationDataset(train_encodings, train_labels)
val_dataset = T5ClassificationDataset(val_encodings, val_labels)

# 5. 训练模型
model = AutoModelForSeq2SeqLM.from_pretrained('/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/flan-t5-base')

training_args = Seq2SeqTrainingArguments(
    #fp8=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    save_total_limit=2,
    evaluation_strategy="epoch",  
    save_strategy="epoch",  
    do_train=True,
    do_eval=True,
    output_dir=f'./results/QuaSiKS_entity_{positive_sample_size}pos',
    load_best_model_at_end=True,
    #metric_for_best_model="precision_for_yes",
    #greater_is_better=True,
    #dataloader_num_workers=4,
    eval_accumulation_steps=1,
    predict_with_generate=True,
    learning_rate=5e-5,
    lr_scheduler_type='cosine',
    weight_decay=0.01,
    warmup_ratio=0.1,
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()