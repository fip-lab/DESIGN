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
parser.add_argument('--top_k_shot', type=int, default=1, help='Top k shot')
parser.add_argument('--model_name', type=str, default='bart-large', help='Model name')
parser.add_argument('--select_algorithm', type=str, default='SELECT', help='Select algorithm')
parser.add_argument('--withabsa', type=bool, default=False, help='With absa')


args = parser.parse_args()


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def save_data(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


tokenizer = BartTokenizer.from_pretrained(f'/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/{args.model_name}')
model = BartForConditionalGeneration.from_pretrained(f'/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/{args.model_name}')
similarity_model = SentenceTransformer('/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/all-MiniLM-L6-v2')
##加载资源向量库

resourse_vector = np.load('/home/zhuangjt/zhuangjt_disk3/SK-TOD/DSTC10_data/resourse/resource_vector_base.npy')
dimension = resourse_vector.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(resourse_vector)

resourse_labels = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/DSTC10_data/resourse/labels.json')
resourse_logs = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/DSTC10_data/resourse/logs.json')

resourse_labels_new = []
resourse_logs_new = []
resourse_logs_aspect_new = []
for label,log in zip(resourse_labels,resourse_logs):
    if label["target"] == True:
        resourse_labels_new.append(label)
        resourse_logs_new.append(log)
        

knowledge_base= load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/DSTC10_data/knowledge.json")

labels_train = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/DSTC10_data/train_split/labels.json")
logs_train = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/DSTC10_data/train_split/logs.json")
labels_val = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/DSTC10_data/val/labels.json")
logs_val = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/DSTC10_data/val/logs.json")

top_k_shot = args.top_k_shot


def create_few_shot(user_query):
    if args.withabsa:
    #definition = "Definition:The output is a response to user query based on provided knowledge."
        definition = "Definition: The output is a response to the user's query based on the provided knowledge. Note that after each review, I provide the aspects and sentiments of the review in the format ['aspect': 'sentiment'; 'aspect': 'sentiment']."
    else:
        definition = "Definition: The output is a response to the user's query based on the provided knowledge."
        

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
    

def create_knowledge_sentence(knowledge_list):
    knowledge_sentence = ""
    for knowledge in knowledge_list:
        question = knowledge_base[knowledge["domain"]][str(knowledge["entity_id"])]["docs"][str(knowledge["doc_id"])]["title"]
        answer = knowledge_base[knowledge["domain"]][str(knowledge["entity_id"])]["docs"][str(knowledge["doc_id"])]["body"]
        knowledge_sentence += "Q: " + question + "\n" + "A: " + answer + "\n"
    return knowledge_sentence[:-1]
                


def create_dataset(labels_data, logs_data):
    dataset = []
    for label ,log in tqdm(zip(labels_data, logs_data),desc="Dataset creating..."):
        if label["target"] == False:
            continue
        else:
            datadict = {}
            query = log[-1]["text"]
            few_shot_text = create_few_shot(query)
            knoledge_text = create_knowledge_sentence(label["knowledge"])
            user_query_text = """Now complete the following example-
Knowledge: 
{}
Query:{}
Answer:""".format(knoledge_text,query)
            datadict["source"] = few_shot_text + user_query_text
            datadict["target"] = label["response"]

            dataset.append(datadict)
    return dataset
    #dataset是一个列表，每个元素是一个字典，包含source和target两个键值对

def list_to_dict(list_data):
    # 初始化一个字典，键为'source'和'target'，值为空列表
    dict_data = {"source": [], "target": []}
    # 遍历列表，分别将'source'和'target'的值追加到相应的列表中
    for item in list_data:
        dict_data["source"].append(item["source"])
        dict_data["target"].append(item["target"])
    return dict_data



train_data  = create_dataset(labels_train, logs_train)
val_data = create_dataset(labels_val, logs_val)
# 将这些列表转换为 Dataset 对象

# 将列表转换为字典
train_data_dict = list_to_dict(train_data)
val_data_dict = list_to_dict(val_data)

train_dataset = Dataset.from_dict(train_data_dict)
val_dataset = Dataset.from_dict(val_data_dict)

# 为了后续使用方便，您可以将这些 Dataset 对象组合成一个 DatasetDict
datasets = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
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

if args.withabsa and args.select_algorithm == 'RANDOM':
    output_path = f"./results/dstc10/{args.model_name}-{args.top_k_shot}-shot-Random-earlystop"
elif args.withabsa and args.select_algorithm == 'SELECT':
    output_path = f"./results/dstc10/{args.model_name}-{args.top_k_shot}-shot-earlystop-absa"
elif not args.withabsa and args.select_algorithm == 'RANDOM':
    output_path = f"./results/dstc10/{args.model_name}-{args.top_k_shot}-shot-withoutabsa-Random-earlystop"
elif not args.withabsa and args.select_algorithm == 'SELECT':
    output_path = f"./results/dstc10/{args.model_name}-{args.top_k_shot}-shot-withoutabsa-earlystop"

training_args = Seq2SeqTrainingArguments(
    output_dir= output_path,                 # 输出目录
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



# 注意：warmup_steps和warmup_ratio同时设置时，只有一个会生效。在这个示例中，我们使用warmup_ratio。
class LossCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_log:
            control = TrainerControl()
            # 计算平均损失值
            avg_loss = sum(state.log_history[-state.num_batches["train"]:]["loss"]) / state.num_batches["train"]
            # 打印平均损失值
            print(f"Epoch {state.epoch}: Average Loss = {avg_loss:.4f}")
# 创建回调实例
loss_callback = LossCallback()
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()