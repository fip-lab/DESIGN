from itertools import combinations
import json
from tqdm import tqdm

def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data
'''
def write_dict(path, data):
    #data是一个字典
'''

def calculate_cooccurrences(data, k):
    cooccurrences = {}

    for knowledges in data:
        if knowledges["target"] == True:  # Only proceed if 'target' is True
            knowledge = knowledges["knowledge"]
            # Convert the knowledge items to the specified string format
            knowledge_strs = [
                f"{item['domain']}__{item['entity_id']}__{item['doc_id']}-{item['sent_id']}"
                if item["doc_type"] == "review" # Use the review ID if the document is a review
                else f"{item['domain']}__{item['entity_id']}__{item['doc_id']}"
                for item in knowledge
            ]

            # Generate all possible pairs without repetition
            for pair in combinations(knowledge_strs, k):
                # Sort the pair to prevent duplicate entries (e.g., A-B, B-A)
                sorted_pair = tuple(sorted(pair))
                cooccurrences[sorted_pair] = cooccurrences.get(sorted_pair, 0) + 1
    return cooccurrences


data = load_data('data/train/labels.json')

k_list = [2,3,4,5]

for number in k_list:
    cooccurrences = calculate_cooccurrences(data,number)
    #统计共现次数的最大值，最小值，平均值，中位数，众数
    cooccurrences_count = list(cooccurrences.values())
    print(f"Number of cooccurrences: {len(cooccurrences)}")
    print(f"Max cooccurrence count: {max(cooccurrences_count)}")
    print(f"Min cooccurrence count: {min(cooccurrences_count)}")
    print(f"Average cooccurrence count: {sum(cooccurrences_count) / len(cooccurrences_count)}")
    print(f"Median cooccurrence count: {sorted(cooccurrences_count)[len(cooccurrences_count) // 2]}")
    print(f"Mode cooccurrence count: {max(set(cooccurrences_count), key=cooccurrences_count.count)}")

    #按照共现次数从大到小排序
    cooccurrences = dict(sorted(cooccurrences.items(), key=lambda item: item[1], reverse=True))

    #write_dict('data/cooccurrences.json', cooccurrences)
    ## 保存共现次数大于等于5的共现关系,保存为csv文件
    with open(f'data/cooccurrences/cooccurrences_{number}.csv', 'w') as f:
        for k, v in tqdm(cooccurrences.items()):
            #根据number的值，判断共现关系中的知识点个数
            if number == 2:   
                    f.write(f"{k[0]},{k[1]},{v}\n")
            elif number == 3:    
                    f.write(f"{k[0]},{k[1]},{k[2]},{v}\n")
            elif number == 4:   
                    f.write(f"{k[0]},{k[1]},{k[2]},{k[3]},{v}\n")
            elif number == 5:
                    f.write(f"{k[0]},{k[1]},{k[2]},{k[3]},{k[4]},{v}\n")
            else:
                print('number的值不在2-5之间')
                break
