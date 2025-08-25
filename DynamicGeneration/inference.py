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

from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from transformers import pipeline


SPECIAL_TOKENS = {
    "additional_special_tokens": ["<user>", "<system>","<knowledge_sep>", "<review>", "<faq>", "<entity>"],
}
user_token = SPECIAL_TOKENS["additional_special_tokens"][0]
system_token = SPECIAL_TOKENS["additional_special_tokens"][1]
knowledge_sep_token = SPECIAL_TOKENS["additional_special_tokens"][2]
review_token = SPECIAL_TOKENS["additional_special_tokens"][3]
faq_token = SPECIAL_TOKENS["additional_special_tokens"][4]
entity_token = SPECIAL_TOKENS["additional_special_tokens"][5]

##导入数据集文件
def load_data(path):
    with open(path,"r",encoding="utf-8") as f:
        data = json.load(f)
    return data

def create_knowledge_sentence(knowledge_organization_mode, entity, knowledge_list,knowledge_with_absa,with_absa):
    print(with_absa)
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





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knowledge_organization_mode", choices=['orginal', 'by_emotion', 'by_reviewer'])
    parser.add_argument("--with_absa", type=str, default="True", choices=["True", "False"])
    parser.add_argument("--dataset_type", choices=['val', 'test'])
    parser.add_argument("--ks_labels_path", type=str, default="ks_labels.json")
    parser.add_argument("--output_path", type=str, default="output.json")
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()

    print(args.with_absa)

    if args.dataset_type == "val":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:1")

    if args.with_absa == "True":
        model_path = args.checkpoint
    else:
        model_path = args.checkpoint
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    #generator = pipeline('text2text-generation', model=model_path, tokenizer=model_path, device=1)

    knowledge_with_absa = load_data("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/knowledge_with_absa.json")
    logs_data = load_data(f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/{args.dataset_type}/logs.json")
    ks_labels_data = load_data(f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/{args.ks_labels_path}")
    if args.with_absa == "True":
        instruct = "Please generate a response based on the user's dialogue history and the related knowledge (reviews/FAQs) of past users. Note that I have provided the aspect and sentiment of each comment after the review, in the format ['aspect': 'sentiment'; 'aspect': 'sentiment']. \n"
    else:
        instruct = "Please generate a response based on the user's dialogue history and the related knowledge (reviews/FAQs) of past users. \n"
    for ks_label, log in tqdm(zip(ks_labels_data, logs_data)):
        if ks_label["target"] == False:
            continue
        else:
            if len(log) > 5:
                log = log[-5:]
            dialog_sentence = 'Dialogue:\n'
            for i, sentence in enumerate(log):
                if i % 2 == 0:  # even index
                    dialog_sentence += "USER:" + sentence["text"] + "\n"
                else:  # odd index
                    dialog_sentence += "SYSTEM:" + sentence["text"] + "\n"
            dialog_sentence +="SYSTEM:"

            knowledge_sentence = "Knowledge: \n"
            #knowledge_sentence += f"{entity} Hotel/Restaurant: "
            knoeldege_list = ks_label["knowledge"]
            entity_set = set()
            for knowledge in knoeldege_list:
                entity_set.add(f"{knowledge['domain']}_{knowledge['entity_id']}")
            for entity in entity_set:
                entity_name = knowledge_with_absa[entity.split("_")[0]][str(entity.split("_")[1])]["name"]
                knowledge_sentence += f"Hotel/Restaurant name: {entity_name}\n"
                knowledge_sentence += create_knowledge_sentence(args.knowledge_organization_mode,entity,knoeldege_list,knowledge_with_absa,args.with_absa)

            #input_sentence = instruct + knowledge_sentence + dialog_sentence
            input_sentence = knowledge_sentence + dialog_sentence
            padding = "max_length"
            input_encodeings = tokenizer(input_sentence, return_tensors="pt", truncation=True, padding=padding, max_length=512)
            input_encodeings = input_encodeings.to(device)
            output_sequences = model.generate(
                input_encodeings["input_ids"],
                attention_mask=input_encodeings["attention_mask"],
                num_beams=5,
                max_length=512,
                early_stopping=True,
                use_cache=True,
            )
            output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
            
            #print(generator(input_sentence, max_length=512, num_beams=5, early_stopping=True, use_cache=True)[0])
            #output_text = generator(input_sentence, max_length=512, num_beams=5, early_stopping=True, use_cache=True)[0]["generated_text"]
            ks_label["response"] = output_text
            
            print("--------------------------------------------------------------")
            #print(input_sentence)
            print('\n')
            #print(output_text)
            
    

    with open(args.output_path,"w",encoding="utf-8") as f:
        json.dump(ks_labels_data,f,ensure_ascii=False,indent=4)



if __name__ == "__main__":
    main()