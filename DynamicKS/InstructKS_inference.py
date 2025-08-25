import json
import random
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from scripts.dataset_walker import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader

knowledge_reader = KnowledgeReader('./data', 'knowledge_with_absa.json')
device = torch.device("cuda")

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained('./InstructKS/results/2pos2neg/checkpoint-35035')  # 指向保存 tokenizer 的目录
model = AutoModelForSeq2SeqLM.from_pretrained('./InstructKS/results/2pos2neg/checkpoint-35035')  # 指向保存模型的目录
model.to(device)

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

'''
# 准备输入数据
input_text = "你的输入文本放在这里"
input_encodings = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# 推断
output_sequences = model.generate(
    input_ids=input_encodings['input_ids'],
    attention_mask=input_encodings['attention_mask'],
    max_length=512,
)

# 解码输出
output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
print(output_text)
'''
dataset_split = "val"

with open(f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/{dataset_split}/em/em.json", 'r') as f:
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



def instructks(knowledge_sent, user_query, aspect_item_of_query):
    sentence = "[user_query]" + user_query + "\n[aspect_item_of_query]" + aspect_item_of_query + "\n[knowledge]" + knowledge_sent
    input_text = definition + sentence + end_instruction
    input_encodings = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_encodings = input_encodings.to(device)
    output_sequences = model.generate(
        input_ids=input_encodings['input_ids'],
        attention_mask=input_encodings['attention_mask'],
        max_length=512,
    )
    output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    if output_text == "yes":
        return True
    else:
        return False

save_interval = 50  # Save after every 50 iterations
save_path = f"/home/zhuangjt/zhuangjt_disk3/SK-TOD/pred/{dataset_split}/instructks/instructks_2pos2neg.json"

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