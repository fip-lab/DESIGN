import json
from tqdm import tqdm 

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

for datatype in ["train","val","test"]:
    print(type(datatype))
    em_flie = load_json(f"./{datatype}/em.json")
    logs_aspect_flie = load_json(f"./{datatype}/logs_aspect.json")
    list = []
    for em, log_aspect in tqdm(zip(em_flie, logs_aspect_flie)):
        if em["target"] == False:
            list.append("")
        else:
            knowledge_keys = em["knowledge"]
            aspect = log_aspect
            sentence = ""
            for knowledge_key in knowledge_keys:
                domain = knowledge_key["domain"]
                entity = knowledge_key["entity_name"]
                sentence += f"{domain}--{entity}--{aspect}; "
            list.append(sentence)
    with open(f"./{datatype}/triple.json", "w") as f:
        json.dump(list, f, indent=4, ensure_ascii=False)
                
            
