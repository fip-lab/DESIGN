import json
from tqdm import tqdm
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import faiss
from transformers import (
    DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,pipeline
)

#from scripts.knowledge_reader import KnowledgeReader
from nltk.translate.bleu_score import corpus_bleu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 如果你的两张卡的编号是0和1


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

#knowledge_reader = KnowledgeReader("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data","knowledge_with_absa.json")


top_k_shot = 3
sum_num = 6

def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def save_data(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def create_knowledge_sentence_for_entity(entity_knowledge_list,do_summary):
    knowledge_sentence = ""
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
        knowledge_sentence += doc_knowledge_sentence + "\n"
        knowledge_sentence = knowledge_sentence
    if do_summary == True:
        knowledge_sentence = summarizer(knowledge_sentence)[0]['summary_text'] + "\n"
    else:
        knowledge_sentence = knowledge_sentence
    return knowledge_sentence[:-1]


def create_knowledge_sentence(knowledge_list):
    #对knowledge_list进行排序，faq排在前面，按照entity_id,doc_id,sent_id排序
    knowledge_list = sorted(knowledge_list, key=lambda x: (x['doc_type'] != 'faq',x['entity_id'], x['doc_id'], x.get('sent_id', 0)))

    if len(knowledge_list) > sum_num:
        do_summary = True
    else:
        do_summary = False

    knowledge_sentence = ""
    entity_set = set()
    for knowledge in knowledge_list:
        entity_set.add(f"{knowledge['domain']}_{knowledge['entity_id']}")
    for entity in entity_set:
        entity_name = knowledge_with_absa[entity.split("_")[0]][str(entity.split("_")[1])]["name"]
        knowledge_sentence += f"{entity_name}\n"
        entity_knowledge_list = [knowledge_snippet for knowledge_snippet in knowledge_list if f"{knowledge_snippet['domain']}_{knowledge_snippet['entity_id']}" == entity]
        
        knowledge_sentence += create_knowledge_sentence_for_entity(entity_knowledge_list,do_summary) + "\n"
        #去掉最后一个换行符
    knowledge_sentence = knowledge_sentence[:-1]
    return knowledge_sentence
                


def create_few_shot(user_query):
    definition = "Definition:The output is a response to user query based on provided knowledge."
    query_embedding = similarity_model.encode([user_query])
    D, I = index.search(query_embedding, k=top_k_shot)
    
    few_shot_text = ""
    for num,idx in enumerate(I[0]):
        resource_label = resourse_labels_new[idx]
        resource_log = resourse_logs_new[idx]
        query = resource_log[-1]["text"]
        knowledge = resource_label["knowledge"]
        response = resource_label["response"]
        knowledge_sentence = create_knowledge_sentence(knowledge)

        text = """Example {}-
input:
user query: {}
knowledge: 
{}
output:{}""".format(num+1,query,knowledge_sentence,response)
        few_shot_text += text + "\n"
    return definition + "\n" + few_shot_text


def create_input_text(user_query,knowledge):
    few_shot_text = create_few_shot(user_query)

    knowledge_sentence = create_knowledge_sentence(knowledge)

    user_query_text = """Now complete the following example-
input:
user query: {}
knowledge: 
{}
output:""".format(user_query,knowledge_sentence)
    input_text = few_shot_text + user_query_text
    #print(input_text)
    #以空格为单位切分
    text = input_text.split(" ")
    #print(len(input_text))
    #print(len(text))
    #print("--------------------------------------------------")
    return input_text
    
def process_data(labels_set,logs_set):
    inputs = []
    labels = []
    for label,log in tqdm(zip(labels_set,logs_set),disable=False,desc='process data'):
        if label["target"] == False:
            continue
        else:
            user_query = log[-1]["text"]
            knowledge = label["knowledge"]
            input_text = create_input_text(user_query,knowledge)
            
            inputs.append(input_text)
            labels.append(label["response"])
    return inputs,labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # 对预测和标签进行解码
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)


    # 计算 BLEU 分数
    bleu_score = corpus_bleu([[ref.split()] for ref in decoded_labels], [pred.split() for pred in decoded_preds])

    # 计算 METEOR 分数
    #meteor_score_avg = np.mean([meteor_score([ref], pred) for pred, ref in zip(decoded_preds, decoded_labels)])

    # 将结果合并
    result = {
        'bleu': round(bleu_score * 100, 4),
    }
    
    return result

class T5ClassificationDataset(Dataset):
    def __init__(self, encodings, labels,padding):
        self.encodings = encodings
        self.labels = labels
        self.padding = "max_length"
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
        response_label = tokenizer(self.labels[idx], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        if self.padding == "max_length":
            response_label["input_ids"] = [(l if l != tokenizer.pad_token_id else -100) for l in response_label["input_ids"][0]]
        
        item['labels'] = response_label["input_ids"]
        #item['labels'] = torch.tensor(tokenizer(self.labels[idx], padding="max_length", truncation=True, max_length=512, return_tensors="pt").input_ids[0])
        return item

    def __len__(self):
        return len(self.labels)


similarity_model = SentenceTransformer('/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/all-MiniLM-L6-v2')
#summarizer = pipeline("summarization", model="/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/MEETING_SUMMARY")

# huggingface hub model id
model_id="/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/flan-t5-large"
# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

##加载资源向量库
resourse_vector = np.load('/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/resourse/resource_vector_base.npy')
dimension = resourse_vector.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(resourse_vector)

resourse_labels = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/resourse/labels.json')
resourse_logs = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/resourse/logs.json')
resourse_logs_aspect = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/resourse/logs_aspect.json')
knowledge_with_absa = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/knowledge_with_absa.json")

resourse_labels_new = []
resourse_logs_new = []
resourse_logs_aspect_new = []
for label,log,log_aspect in zip(resourse_labels,resourse_logs,resourse_logs_aspect):
    if label["target"] == True:
        resourse_labels_new.append(label)
        resourse_logs_new.append(log)
        resourse_logs_aspect_new.append(log_aspect)
'''
train_labels_set = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/train/labels.json')
train_logs_set = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/train/logs.json')

val_labels_set = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/val/labels.json')
val_logs_set = load_data('/home/zhuangjt/zhuangjt_disk3/SK-TOD/FlanGeneration/Dynamic_Dataset/val/logs.json')

train_inputs, train_labels = process_data(train_labels_set,train_logs_set)
val_inputs, val_labels = process_data(val_labels_set,val_logs_set)
'''
train_inputs = load_data("./Dynamic_Dataset/train_inputs.json")
train_labels = load_data("./Dynamic_Dataset/train_labels.json")
val_inputs = load_data("./Dynamic_Dataset/val_inputs.json")
val_labels = load_data("./Dynamic_Dataset/val_labels.json")

'''
save_data(val_inputs,"./Dynamic_Dataset/val_inputs.json")
save_data(val_labels, "./Dynamic_Dataset/val_labels.json")

save_data(train_inputs,"./Dynamic_Dataset/train_inputs.json")
save_data(train_labels, "./Dynamic_Dataset/train_labels.json")
'''
padding = "max_length"
train_encodings = tokenizer(train_inputs, truncation=True, padding=padding, max_length=512)
val_encodings = tokenizer(val_inputs, truncation=True, padding=padding, max_length=512)

train_dataset = T5ClassificationDataset(train_encodings, train_labels, padding)
val_dataset = T5ClassificationDataset(val_encodings, val_labels, padding)

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(tokenizer, 
                                        model=model,
                                        label_pad_token_id=label_pad_token_id,
                                        pad_to_multiple_of=8)
#tensorboard_callback = TensorBoardCallback()


training_args = Seq2SeqTrainingArguments(
    fp16=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    evaluation_strategy="epoch",  
    save_strategy="epoch",  
    #save_total_limit=5,
    do_train=True,
    do_eval=True,
    output_dir=f"./dinstruct_gen",
    #output_dir=f'./results/DInstructKS-flan-{alignment_type}-neg{sample_size}pos{sample_size}-reviewer',
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    #dataloader_num_workers=4,
    eval_accumulation_steps=1,
    predict_with_generate=True,
    learning_rate=5e-5,
    lr_scheduler_type='cosine',
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_dir='./logs/dinstruct_gen',
    report_to="tensorboard",
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    #callbacks=[tensorboard_callback], 
)

trainer.train()