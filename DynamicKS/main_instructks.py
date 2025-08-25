import json
import random
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
#from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 如果你的两张卡的编号是0和1


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 1. 加载数据
with open('./InstructKS/data/both/train_dataset.json', 'r') as file:
    train_data = json.load(file)
    
with open('./InstructKS/data/both/val_dataset.json', 'r') as file:
    val_data = json.load(file)

#对train_data 进行采样，选取所有labels为yes的数据，然后随机采样与正样本数量相同的负样本(labels为no)
yes_data = []
no_data = []
for item in train_data:
    if item['labels'] == 'yes':
        yes_data.append(item)
    else:
        no_data.append(item)
sampled_no_data = random.sample(no_data, len(yes_data))

train_data = yes_data + sampled_no_data

yes_data = []
no_data = []
for item in val_data:
    if item['labels'] == 'yes':
        yes_data.append(item)
    else:
        no_data.append(item)
sampled_no_data = random.sample(no_data, 2*len(yes_data))

val_data = yes_data + sampled_no_data

random.shuffle(train_data)
random.shuffle(val_data)
##训练集和验证集只取原来的一半
train_data = train_data[:int(len(train_data)/2)]
val_data = val_data[:int(len(val_data)/4)]

print(len(train_data))
print(len(val_data))

class T5ClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(tokenizer(self.labels[idx], padding="max_length", truncation=True, max_length=512, return_tensors="pt").input_ids[0])
        return item

    def __len__(self):
        return len(self.labels)

# 定义一个评估函数
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


# 2. 数据预处理
definition = """Definition: The output will be an assessment of whether the knowledge can be used to support answering the user's inquiry.If it can support, the output will be "yes"; otherwise, the output will be "no".
Positive example 1-
input:
[user_query]Does this hotel have rooms with a good view of the neighborhood?
[aspect_item_of_query]rooms, view.
[knowledge]There was a nice size refrigerator and a beautiful  view out the window of the 7th floor. 
output:yes
Positive example 2-
input:
[user_query]Does Nandos happen to serve beer?
[aspect_item_of_query]beer
[knowledge]question:Does Nandos have alcoholic drinks? answer:Nando's has alcohol drinks.
output:yes
Negative example 1-
input:
[user_query]Which one has the least noisy, disruptive environment?
[aspect_item_of_query]quality of beer served
[knowledge]If you're looking for a cool, hip Spanish restaurant to visit for tapas and drinks, I wouldn't recommend La Raza because their drinks were frankly not well-prepared and tasted pretty mediocre.
output:no
Negative example 2-
input:
[user_query]Are the portion sizes here large?
[aspect_item_of_query]portion sizes
[knowledge]question: Do you have any type of seating for babies? answer: The Golden Curry has high chairs for infants.
output:no
Now complete the following example-
input:
"""  # 在这里定义你的 task prefix
end_instruction = ' \noutput:'

def process_data(data):
    inputs = []
    labels = []
    for item in data:
        inputs.append(definition + item['text'] + end_instruction)
        labels.append(item['labels'])
    return inputs, labels

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
    #fp16=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    evaluation_strategy="epoch",  
    save_strategy="epoch",  
    do_train=True,
    do_eval=True,
    output_dir='./results/Instruct-flan-2pos2neg',
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
    #compute_metrics=compute_metrics,
)

trainer.train()



