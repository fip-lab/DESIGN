import json
import random
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader

knowledge_reader = KnowledgeReader('./data', 'knowledge_with_absa.json')
device = torch.device("cuda:1")

SPECIAL_TOKENS = {
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>","<faq_tag>", "<review_tag>","<user_query>","<aspect item>"],
}

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained('./runs/ks-deberta-v3-large-both-oracle')  # 指向保存 tokenizer 的目录
model = AutoModelForSequenceClassification.from_pretrained('./runs/ks-deberta-v3-large-both-oracle')  # 指向保存模型的目录
model.to(device)
tokenizer.add_special_tokens(SPECIAL_TOKENS)


both_knowledge_prompt = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Whether the faq or review matches the user query and its aspect items. 1 is a match, 0 is not a match"))
cls = tokenizer.cls_token_id
sep = tokenizer.sep_token_id
pad = tokenizer.pad_token_id
speaker1, speaker2, knowledge_sep, knowledge_tag, faq_tag, review_tag, user_query_tag, aspect_item_tag = tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS["additional_special_tokens"]
        )

dataset_split = "test"

with open(f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/{dataset_split}/em/em_deberta-v3-base.json", 'r') as f:
    em_knowledge = json.load(f)

with open(f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/{dataset_split}/logs.json", 'r') as f:
    logs = json.load(f)

with open(f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/{dataset_split}/logs_aspect.json", 'r') as f:
    logs_aspect = json.load(f)

with open("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/vocab_dish_or_drink.json",'r') as f:
    vocab_dish_or_drink = json.load(f)

'''
对应special dish/drink的逻辑,先确定em中实体domain,如果是“restaurant”,
根据“entity_id”获取“entity_name”,然后到vocab_dish_or_drink取对应的列表,
然后对user_query做模糊匹配,如果匹配到对应的菜式（可能有多个）,用list保存对应菜式名称.
然后只选取包含对应菜式的用户评价来做知识选择
'''
def specific_dish_or_drink(user_query, vocab_list):
    is_specific_dish_or_drink = False
    matched_entities = []
    for word in vocab_list:
        # 直接匹配
        if word.lower() in user_query.lower() and word not in matched_entities:
            matched_entities.append(word)
    if matched_entities:
        is_specific_dish_or_drink = True
    return is_specific_dish_or_drink, matched_entities


def pad_ids(arrays, padding, max_length=-1):
    if max_length < 0:
        max_length = max(list(map(len, arrays)))

    arrays = [
        array + [padding] * (max_length - len(array))
        for array in arrays
    ]

    return arrays

def instructks(knowledge_sent, user_query, aspect_item_of_query):
    aspect_sentence = "The aspect item of query is " + aspect_item_of_query + "."
    aspect_sentence_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(aspect_sentence))
    knowledge_sent_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(knowledge_sent))
    user_query_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(user_query))

    sequence0 = [cls] + [user_query_tag] + user_query_tokenized + [sep] + [aspect_item_tag] + aspect_sentence_tokenized + [sep] + [review_tag] + knowledge_sent_tokenized + [sep]
    sequence1 = both_knowledge_prompt + [sep]
    input_encodings = {}
    input_encodings["input_ids"] = sequence0 + sequence1
    input_encodings["token_type_ids"] = [0 for _ in sequence0] + [1 for _ in sequence1]
    

    input_encodings["input_ids"] = torch.tensor(pad_ids([input_encodings["input_ids"]], pad, max_length=512))
    input_encodings["attention_mask"] = 1-(input_encodings["input_ids"] == pad).int()
    input_encodings["token_type_ids"] = torch.tensor(pad_ids([input_encodings["token_type_ids"]], pad, max_length=512))

    
    '''
    input_encodings = tokenizer(input_encodings, return_tensors="pt", padding=True, truncation=True, max_length=512)
    '''
    #input_encodings = input_encodings.to(device)
    output_sequences = model.generate(
        input_ids=input_encodings['input_ids'],
        attention_mask=input_encodings['attention_mask'],
        max_length=512,
    )
    output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    if output_text == 1:
        return True
    else:
        return False

save_interval = 500  # Save after every 50 iterations
save_path = f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/{dataset_split}/deberta-large-both-special.json"

list = []
##加载emm_knowledge,logs,logs_aspect
for idx,(em, log, log_aspect) in enumerate(tqdm(zip(em_knowledge, logs, logs_aspect))):
    dict = {}
    if em["target"] == False:
        dict["target"] = False
        list.append(dict)
        continue
    else:
        dict["target"] = True
        user_query = log[-1]["text"]
        aspect_item_of_query = log_aspect
        
        ##获取实体的domain和entity_id
        knowledge_infs = em["knowledge"]
        knowledge_list = []
        for knowledge_info in knowledge_infs:  
            domain = knowledge_info["domain"]
            entity_id = knowledge_info["entity_id"]
            if domain == 'restaurant':
                entity_name = knowledge_reader.get_entity_name(domain, entity_id)
                vocab_list = vocab_dish_or_drink[entity_name]
                is_specific_dish_or_drink, matched_entities = specific_dish_or_drink(user_query, vocab_list)
                if is_specific_dish_or_drink:
                    for review_doc_id in knowledge_reader.get_review_doc_ids(domain, entity_id):
                        review_doc = knowledge_reader.get_review_doc(domain, entity_id, review_doc_id)
                        for match_entity in matched_entities:
                            if (match_entity in review_doc['drinks']) or (match_entity in review_doc['dishes']):
                                for review_sent_id, review_sent in review_doc['sentences'].items():
                                    knowledge_sent = review_sent
                                    response = instructks(knowledge_sent, user_query, aspect_item_of_query)
                                    if response == True:
                                        dict_know = {}
                                        dict_know["domain"] = domain
                                        dict_know["entity_id"] = entity_id
                                        dict_know["doc_type"] = "review"
                                        dict_know["doc_id"] = int(review_doc_id)
                                        dict_know["sent_id"] = int(review_sent_id)
                                        knowledge_list.append(dict_know)
                else:
                    for review_doc_id in knowledge_reader.get_review_doc_ids(domain, entity_id):
                        review_doc = knowledge_reader.get_review_doc(domain, entity_id, review_doc_id)
                        for review_sent_id, review_sent in review_doc['sentences'].items():
                            knowledge_sent = review_sent
                            response = instructks(knowledge_sent, user_query, aspect_item_of_query)
                            if response == True:
                                dict_know = {}
                                dict_know["domain"] = domain
                                dict_know["entity_id"] = entity_id
                                dict_know["doc_type"] = "review"
                                dict_know["doc_id"] = int(review_doc_id)
                                dict_know["sent_id"] = int(review_sent_id)
                                knowledge_list.append(dict_know)
            
            else:
                ##获取knowledge
                for review_doc_id in knowledge_reader.get_review_doc_ids(domain, entity_id):
                    review_doc = knowledge_reader.get_review_doc(domain, entity_id, review_doc_id)
                    for review_sent_id, review_sent in review_doc['sentences'].items():
                        knowledge_sent = review_sent
                        response = instructks(knowledge_sent, user_query, aspect_item_of_query)
                        if response == True:
                            dict_know = {}
                            dict_know["domain"] = domain
                            dict_know["entity_id"] = entity_id
                            dict_know["doc_type"] = "review"
                            dict_know["doc_id"] = int(review_doc_id)
                            dict_know["sent_id"] = int(review_sent_id)
                            knowledge_list.append(dict_know)
            for faq_doc_id in knowledge_reader.get_faq_doc_ids(domain, entity_id):
                faq_doc = knowledge_reader.get_faq_doc(domain, entity_id, faq_doc_id)
                knowledge_sent = f"question:{faq_doc['question']} answer:{faq_doc['answer']}"
                response = instructks(knowledge_sent, user_query, aspect_item_of_query)
                if response == True:
                    dict_know = {}
                    dict_know["domain"] = domain
                    dict_know["entity_id"] = entity_id
                    dict_know["doc_type"] = "faq"
                    dict_know["doc_id"] = int(faq_doc_id)
                    knowledge_list.append(dict_know)
        dict["knowledge"] = knowledge_list
        dict["response"] = "none"

    list.append(dict)

    if (idx + 1) % save_interval == 0 or (idx + 1) == len(em_knowledge):
        with open(save_path, 'w') as f:
            json.dump(list, f, indent=4)
'''
with open(f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/{dataset_split}/instructks/instructks_quarter.json", 'w') as f:
    json.dump(dict, f, indent=4)
'''                  