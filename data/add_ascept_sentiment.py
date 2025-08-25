from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import json

device = ("cuda:1")

ate_tokenizer = AutoTokenizer.from_pretrained("PLM/ate_tk-instruct-base-def-pos-neg-neut-combined")
ate_model = AutoModelForSeq2SeqLM.from_pretrained("PLM/ate_tk-instruct-base-def-pos-neg-neut-combined").to(device)
atsc_tokenizer = AutoTokenizer.from_pretrained("PLM/atsc_tk-instruct-base-def-pos-neg-neut-combined")
atsc_model = AutoModelForSeq2SeqLM.from_pretrained("PLM/atsc_tk-instruct-base-def-pos-neg-neut-combined").to(device)

ate_instruct = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
    Positive example 1-
    input: I charge it at night and skip taking the cord with me because of the good battery life.
    output: battery life
    Positive example 2-
    input: I even got my teenage son one, because of the features that it offers, like, iChat, Photobooth, garage band and more!.
    output: features, iChat, Photobooth, garage band
    Negative example 1-
    input: Speaking of the browser, it too has problems.
    output: browser
    Negative example 2-
    input: The keyboard is too slick.
    output: keyboard
    Neutral example 1-
    input: I took it back for an Asus and same thing- blue screen which required me to remove the battery to reset.
    output: battery
    Neutral example 2-
    input: Nightly my computer defrags itself and runs a virus scan.
    output: virus scan
    Now complete the following example-
    input: """
atsc_instruct ="""Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
    Otherwise, the output should be 'neutral'. For aspects which are classified as noaspectterm, the sentiment is none.
    Positive example 1-
    input: With the great variety on the menu , I eat here often and never get bored. The aspect is menu.
    output: positive
    Positive example 2- 
    input: Great food, good size menu, great service and an unpretensious setting. The aspect is food.
    output: positive
    Negative example 1-
    input: They did not have mayonnaise, forgot our toast, left out ingredients (ie cheese in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it. The aspect is toast.
    output: negative
    Negative example 2-
    input: The seats are uncomfortable if you are sitting against the wall on wooden benches. The aspect is seats.
    output: negative
    Neutral example 1-
    input: I asked for seltzer with lime, no ice. The aspect is seltzer with lime.
    output: neutral
    Neutral example 2-
    input: They wouldnt even let me finish my glass of wine before offering another. The aspect is glass of wine.
    output: neutral
    Now complete the following example-
    input: """

def get_aspect_term(text):
    delim_instruct = ''
    eos_instruct = ' \noutput:'
    tokenized_text = ate_tokenizer(ate_instruct + text + delim_instruct + eos_instruct, return_tensors="pt").to(device)
    output = ate_model.generate(tokenized_text.input_ids)
    result = ate_tokenizer.decode(output[0], skip_special_tokens=True)
    if result == 'noaspectterm':
        return ""
    else:
        return result
    return result

def get_aspect_sentiment(text):
        delim_instruct = ' The aspect is '
        eos_instruct = '.\noutput:'
        aspect_term = get_aspect_term(text)
        ##用列表保存多个aspect
        items = aspect_term.split(',')
        # 去除每个项目的前后空格
        aspect_term = [item.strip() for item in items]
            
        ##用字典保存方面和情感
        aspect_sentiment = {}
        for i in aspect_term:
            tokenized_text = atsc_tokenizer(atsc_instruct + text + delim_instruct + i + eos_instruct, return_tensors="pt").to(device)
            output = atsc_model.generate(tokenized_text.input_ids)
            sentiment = atsc_tokenizer.decode(output[0], skip_special_tokens=True)
            if sentiment == 'positive' or sentiment == 'negative' or sentiment == 'neutral':
                aspect_sentiment[i] = sentiment
        return aspect_sentiment

##先读取数据
'''
knowledge
"0": {
          "traveler_type": "Solo travelers",
          "sentences": {
            "0": "I was really happy with my recent stay at A and B Guest House.",
            "1": "I stayed on my own, and I'm a smoker, so I was super happy that there was a designated area especially for smokers.",
            "2": "I also thought that my room was very spacious, and I was pleased with the breakfast options that were available."
          }
--->
"0": {
          "traveler_type": "Solo travelers",
          "sentences": {
            "0": "I was really happy with my recent stay at A and B Guest House.",
            "1": "I stayed on my own, and I'm a smoker, so I was super happy that there was a designated area especially for smokers.",
            "2": "I also thought that my room was very spacious, and I was pleased with the breakfast options that were available."
          }
          "ascept_sentiment": [
            {"noaspectterm": "neutral"},
            {"smokers": "positive"},
            {"room": "positive",
            "breakfast options": "positive"}
            ]

log不好在里面加新的数据,用另外一个文件保存,一一对应,用的时候zip一下

'''

'''

with open('data/knowledge.json', 'r', encoding='utf-8') as f:
    knowledge_base = json.load(f)

for knowledge_domain in knowledge_base:
     for knowledge_entity in tqdm(knowledge_base[knowledge_domain].values(), desc=f'Processing {knowledge_domain}'):
        reviews = knowledge_entity['reviews']
        #print(reviews)
        faqs = knowledge_entity["faqs"]
        for review in tqdm(reviews.values(), desc=f'Processing  reviews'):
            sentence_dict = review["sentences"]
            
            absa_list = []
            for sentence_id in sentence_dict.keys():
                sentence = sentence_dict[sentence_id]
                aspect_sentiment = get_aspect_sentiment(sentence)
                #aspect_sentiment是一个字典
                absa_list.append(aspect_sentiment)
                #print(aspect_sentiment)
            review["aspect_sentiment"] = absa_list
        for faq in tqdm(faqs.values(), desc=f'Processing  faqs'):
            sentence = f'Question: {faq["question"]} Answer: {faq["answer"]}'
            aspect_sentiment = get_aspect_sentiment(sentence)
            #print(aspect_sentiment)
            faq["aspect_sentiment"] = aspect_sentiment
with open('data/knowledge_with_absa.json', 'w', encoding='utf-8') as f:
    json.dump(knowledge_base, f, indent=4, ensure_ascii=False)


for type in tqdm(["resourse"], desc=f'Processing  logs'):
    with open(f'/home/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/{type}/logs.json', 'r', encoding='utf-8') as f:
        logs = json.load(f)
        ##logs是一个list
    aspect_list = []
    for log in tqdm(logs, desc=f'Processing  log'):
        #log也是一个list,但是list中每一个元素是一个字典，每一个字典是一个对话
        #取最后一个对话
        absa_map = {}
        dialog = log[-1]["text"]
        #print(dialog)
        aspect = get_aspect_term(dialog)
        sentiment = get_aspect_sentiment(dialog)
        absa_map["aspect"] = aspect
        absa_map["sentiment"] = sentiment

        #print(aspect)
        aspect_list.append(absa_map)
    with open(f'/home/zhuangjt/zhuangjt_disk3/SK-TOD/InstructKS/Dynamic_InstructKS_Dataset/{type}/logs_absa.json', 'w', encoding='utf-8') as f:
        json.dump(aspect_list, f, indent=4, ensure_ascii=False)
'''

for type in tqdm(["train"], desc=f'Processing  logs'):
    with open(f'./data/{type}/logs.json', 'r', encoding='utf-8') as f:
        logs = json.load(f)
        ##logs是一个list
    aspect_list = []
    for log in tqdm(logs, desc=f'Processing  log'):
        #log也是一个list,但是list中每一个元素是一个字典，每一个字典是一个对话
        #取最后一个对话
        absa_map = {}
        dialog = log[-1]["text"]
        #print(dialog)
        aspect = get_aspect_term(dialog)
        sentiment = get_aspect_sentiment(dialog)
        absa_map["aspect"] = aspect
        absa_map["sentiment"] = sentiment

        #print(aspect)
        aspect_list.append(absa_map)
    with open(f'data/{type}/logs_absa.json', 'w', encoding='utf-8') as f:
        json.dump(aspect_list, f, indent=4, ensure_ascii=False)