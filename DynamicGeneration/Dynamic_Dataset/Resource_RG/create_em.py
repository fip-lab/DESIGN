from tqdm import tqdm
import json

with open('./labels.json', 'r') as file:
    labels = json.load(file)
with open('../../knowledge.json', 'r') as file:
    knowledge_data = json.load(file)

list = []

for label in labels:
    dict = {}
    if label["target"] == False:
        dict = label
    else:
        dict["target"] = True
        knowledge = []
        # 提取并转换信息，确保唯一性
        unique_entries = set()
        knowledge = []
        for item in label["knowledge"]:
            domain = item["domain"]
            entity_id = item["entity_id"]
            entity_name = knowledge_data[domain][str(entity_id)]["name"]

            unique_key = f"{domain}-{entity_id}-{entity_name}"
            if unique_key not in unique_entries:
                unique_entries.add(unique_key)
                knowledge.append({
                    "domain": domain,
                    "entity_id": entity_id,
                    "entity_name": entity_name
                })
        dict["knowledge"] = knowledge
    list.append(dict)

with open('./em.json','w') as file:
    json.dump(list, file, indent=4)