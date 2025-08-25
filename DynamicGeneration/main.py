import json
import os
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import Dataset
import evaluate
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score

from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,DataCollatorForSeq2Seq,Seq2SeqTrainer, Seq2SeqTrainingArguments
#from transformers.integrations import TensorBoardCallback

# huggingface hub model id
model_id="/home/zhuangjt/zhuangjt_disk3/SK-TOD/PLM/flan-t5-base"
# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

SPECIAL_TOKENS = {
    "additional_special_tokens": ["<user>", "<system>","<knowledge_sep>", "<review>", "<faq>", "<entity>"],
}

tokenizer.add_special_tokens(SPECIAL_TOKENS)
user_token = SPECIAL_TOKENS["additional_special_tokens"][0]
system_token = SPECIAL_TOKENS["additional_special_tokens"][1]
knowledge_sep_token = SPECIAL_TOKENS["additional_special_tokens"][2]
review_token = SPECIAL_TOKENS["additional_special_tokens"][3]
faq_token = SPECIAL_TOKENS["additional_special_tokens"][4]
entity_token = SPECIAL_TOKENS["additional_special_tokens"][5]

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


##导入数据集文件
def load_data(path):
    with open(path,"r",encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_knowledge_sentence(knowledge_organization_mode, entity, knowledge_list,knowledge_with_absa,with_absa):
    ##完成每个entity的knowledge的组织
    knowledge_sentence = ""
    if knowledge_organization_mode == "orginal":
        for knowledge in knowledge_list:
            if f"{knowledge['domain']}_{knowledge['entity_id']}" == entity:
                if knowledge["doc_type"] == "review":
                    sentence = knowledge_with_absa[knowledge["domain"]][str(knowledge["entity_id"])]["reviews"][str(knowledge["doc_id"])]["sentences"][str(knowledge["sent_id"])]
                    aspect_sentiment = knowledge_with_absa[knowledge["domain"]][str(knowledge["entity_id"])]["reviews"][str(knowledge["doc_id"])]["aspect_sentiment"][int(knowledge["sent_id"])]
                    if with_absa == "True":
                        aspect_sentiment_str = "["
                        for aspect in aspect_sentiment.keys():
                            aspect_sentiment_str += f'"{aspect}":"{aspect_sentiment[aspect]}";'
                        aspect_sentiment_str = aspect_sentiment_str[:-1] + "]"
                        knowledge_sentence += f"{review_token} {sentence} {aspect_sentiment_str}\n"
                    else:
                        knowledge_sentence += f"{review_token} {sentence}\n"
                elif knowledge["doc_type"] == "faq":
                    question = knowledge_with_absa[knowledge["domain"]][str(knowledge["entity_id"])]["faqs"][str(knowledge["doc_id"])]["question"]
                    answer = knowledge_with_absa[knowledge["domain"]][str(knowledge["entity_id"])]["faqs"][str(knowledge["doc_id"])]["answer"]
                    knowledge_sentence += f"{faq_token} question: {question}  answer: {answer}\n" 
    
    
    elif knowledge_organization_mode == "by_reviewer":
        reviewer_dict = {}
        for knowledge in knowledge_list:
            if f"{knowledge['domain']}_{knowledge['entity_id']}" == entity:
                if knowledge["doc_type"] == "faq":
                    question = knowledge_with_absa[knowledge["domain"]][str(knowledge["entity_id"])]["faqs"][str(knowledge["doc_id"])]["question"]
                    answer = knowledge_with_absa[knowledge["domain"]][str(knowledge["entity_id"])]["faqs"][str(knowledge["doc_id"])]["answer"]
                    knowledge_sentence += f"{faq_token} question: {question}  answer: {answer}\n" 
                elif knowledge["doc_type"] == "review":
                    key = f"{knowledge['domain']}_{knowledge['entity_id']}_{knowledge['doc_id']}"
                    if key not in reviewer_dict.keys():
                        reviewer_dict[key] = []
                    reviewer_dict[key].append(knowledge["sent_id"])
        
        sentence = ""
        aspect_sentiment_dict = {}
        for key in reviewer_dict.keys():
            #将该key中对应的review按照sent_id从小到大排序
            reviewer_dict[key].sort()
            domain = key.split("_")[0]
            entity_id = key.split("_")[1]
            doc_id = key.split("_")[2]
            for sent_id in reviewer_dict[key]:
                sentence += knowledge_with_absa[domain][str(entity_id)]["reviews"][str(doc_id)]["sentences"][str(sent_id)]
                #将aspect_sentiment_dict中的aspect_sentiment和当前的aspect_sentiment进行合并
                aspect_sentiment = knowledge_with_absa[domain][str(entity_id)]["reviews"][str(doc_id)]["aspect_sentiment"][int(sent_id)]
                aspect_sentiment_dict = {**aspect_sentiment_dict, **aspect_sentiment}
        if with_absa == "True":
            aspect_sentiment_str = "["
            for aspect in aspect_sentiment_dict.keys():
                aspect_sentiment_str += f'"{aspect}":"{aspect_sentiment_dict[aspect]}";'
            aspect_sentiment_str = aspect_sentiment_str[:-1] + "]"
            knowledge_sentence += f"{review_token} {sentence} {aspect_sentiment_str}\n"
        else:
            knowledge_sentence += f"{review_token} {sentence}\n"


    elif knowledge_organization_mode == "by_emotion":
        pass
    
    else:
        raise ValueError("knowledge_organization_mode must be one of 'out of order', 'by_emotion', 'by_reviewer'")
    
    return knowledge_sentence

def create_dataset(labels_data, logs_data, knowledge_with_absa, knowledge_organization_mode, if_knowledge_denoise,with_absa):
    dateset = []
    for label ,log in tqdm(zip(labels_data, logs_data),desc="Dataset creating..."):
        if label["target"] == False:
            continue
        else:
            datadict = {}
            #Due to the model's length limitations, in order to preserve knowledge information, we have truncated the dialogue history.
            if len(log) > 5:
                log = log[-5:]

            ##三个项，分别是dialogue，knowledge，response
            dialog_sentence = 'Dialogue:\n'
            for i, sentence in enumerate(log):
                if i % 2 == 0:  # even index
                    dialog_sentence += "USER:" + sentence["text"] + "\n"
                else:  # odd index
                    dialog_sentence += "SYSTEM:" + sentence["text"] + "\n"
            dialog_sentence +="SYSTEM:"

            
            knowledge_sentence = "Knowledge: \n"
            #knowledge_sentence += f"{entity} Hotel/Restaurant: "
            knoeldege_list = label["knowledge"]
            entity_set = set()
            for knowledge in knoeldege_list:
                entity_set.add(f"{knowledge['domain']}_{knowledge['entity_id']}")
            for entity in entity_set:
                entity_name = knowledge_with_absa[entity.split("_")[0]][str(entity.split("_")[1])]["name"]
                knowledge_sentence += f"Hotel/Restaurant name: {entity_name}\n"
                knowledge_sentence += create_knowledge_sentence(knowledge_organization_mode, entity, knoeldege_list,knowledge_with_absa,with_absa)

            #print(knowledge_sentence)
            
            datadict["dialog"] = dialog_sentence
            datadict["knowledge"] = knowledge_sentence
            datadict["response"] = label["response"]
            dateset.append(datadict)
    return dateset





##模型的输入是defiation+knowledge+dialogue

def process_data(examples,instruct):
    inputs = [instruct + ex["knowledge"] + ex["dialog"] for ex in examples]
    #print(inputs[:20])
    labels = [ex["response"] for ex in examples]
    return inputs, labels



# 加载 ROUGE 指标


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # 对预测和标签进行解码
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
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
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge_organization_mode", choices=['orginal', 'by_emotion', 'by_reviewer'])
    parser.add_argument("--if_knowledge_denoise", action="store_true", default=False)
    parser.add_argument("--with_absa", type=str, default="True", choices=["True", "False"])
    args = parser.parse_args()
   

    print(args.knowledge_organization_mode)
    print(args.if_knowledge_denoise)
    print(args.with_absa)

    knowledge_with_absa_file = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/knowledge_with_absa.json")
    labels_train = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/train/labels.json")
    logs_train = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/train/logs.json")
    labels_val = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/val/labels.json")
    logs_val = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/val/logs.json")

    if args.with_absa == "True":
        instruct = "Please generate a response based on the user's dialogue history and the related knowledge (reviews/FAQs) of past users. Note that I have provided the aspect and sentiment of each comment after the review, in the format ['aspect': 'sentiment'; 'aspect': 'sentiment']. \n"
    else:
        instruct = "Please generate a response based on the user's dialogue history and the related knowledge (reviews/FAQs) of past users. \n"

    train_dataset = create_dataset(labels_train, logs_train, knowledge_with_absa_file, args.knowledge_organization_mode, args.if_knowledge_denoise,args.with_absa)
    val_dataset = create_dataset(labels_val, logs_val, knowledge_with_absa_file, args.knowledge_organization_mode, args.if_knowledge_denoise,args.with_absa)
    print(len(train_dataset))

    train_texts, train_labels = process_data(train_dataset,instruct)
    val_texts, val_labels = process_data(val_dataset,instruct)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)

    train_dataset = T5ClassificationDataset(train_encodings, train_labels)
    val_dataset = T5ClassificationDataset(val_encodings, val_labels)
    
    # we want to ignore tokenizer pad token in the loss
    #label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(tokenizer, 
                                           model=model)
                                           #label_pad_token_id=label_pad_token_id,
                                           #pad_to_multiple_of=8)
    #tensorboard_callback = TensorBoardCallback()


    training_args = Seq2SeqTrainingArguments(
        #fp16=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        evaluation_strategy="epoch",  
        save_strategy="epoch",  
        #save_total_limit=5,
        do_train=True,
        do_eval=True,
        output_dir=f"./checkpoint/{args.knowledge_organization_mode}_{args.with_absa}",
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
        logging_dir='./logs',
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

if __name__ == "__main__":
    main()