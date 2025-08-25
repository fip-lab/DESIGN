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
parser.add_argument('--top_k_shot', type=int, default=3, help='Top k shot')
parser.add_argument('--model_name', type=str, default='bart-base', help='Model name')
parser.add_argument('--withabsa', type=bool, default=True, help='With absa')
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

resourse_vector = np.load('/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/Dynamic_Dataset/resourse/resource_vector_base.npy')
dimension = resourse_vector.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(resourse_vector)

resourse_labels = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/Dynamic_Dataset/resourse/labels.json')
resourse_logs = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/Dynamic_Dataset/resourse/logs.json')
resourse_logs_aspect = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/DynamicGeneration/Dynamic_Dataset/resourse/logs_aspect.json')
resourse_labels_new = []
resourse_logs_new = []
resourse_logs_aspect_new = []
for label,log,log_aspect in zip(resourse_labels,resourse_logs,resourse_logs_aspect):
    if label["target"] == True:
        resourse_labels_new.append(label)
        resourse_logs_new.append(log)
        resourse_logs_aspect_new.append(log_aspect)

knowledge_with_absa = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/knowledge_with_absa.json")
knowledge_with_absa_file = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/knowledge_with_absa.json")
labels_train = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/train/labels.json")
logs_train = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/train/logs.json")
labels_val = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/val/labels.json")
logs_val = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/val/logs.json")

top_k_shot = args.top_k_shot

def create_few_shot(user_query):
    if args.withabsa:
    #definition = "Definition:The output is a response to user query based on provided knowledge."
        definition = "Definition: The output is a response to the user's query based on the provided knowledge. Note that after each review, I provide the aspects and sentiments of the review in the format ['aspect': 'sentiment'; 'aspect': 'sentiment']."
    else:
        definition = "Definition: The output is a response to the user's query based on the provided knowledge."

    few_shot_text = ""

    return definition + "\n" + few_shot_text
    #return few_shot_text
    
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
        #knowledge_sentence += doc_knowledge_sentence + "\n"
        if args.withabsa:
            knowledge_sentence += doc_knowledge_sentence + aspect_sentiment_str + "\n"
        else:
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
            user_query_text = """Complete the following dialogue-
Knowledge: 
{}
Query:{}
Answer:""".format(knoledge_text,query)
            datadict["source"] = user_query_text
            datadict["target"] = label["response"]
            #print(datadict["source"])
            #print(datadict["target"])
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

print(len(train_dataset))
print(len(val_dataset))
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



training_args = Seq2SeqTrainingArguments(
    output_dir= f"./results/{args.model_name}-{args.top_k_shot}-shot-withoutfewshot-withoutabsa-earlystop",                 # 输出目录
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