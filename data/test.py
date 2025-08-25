import json

with open ('knowledge_with_absa.json', 'r') as f:
    knowledge_base = json.load(f)

with open ('train/logs.json', 'r') as f:
    test_log = json.load(f)

with open ('train/logs_aspect.json', 'r') as f:
    test_log_aspect = json.load(f)
with open ('train/labels.json', 'r') as f:
    test_label = json.load(f)

knowledge = knowledge_base["restaurant"][str(19214)]["reviews"][str(2)]["sentences"][str(3)]

print(knowledge)
knowledge = knowledge_base["restaurant"][str(19214)]["reviews"][str(0)]["sentences"][str(7)]
print(knowledge)
knowledge = knowledge_base["restaurant"][str(19214)]["reviews"][str(3)]["sentences"][str(4)]
print(knowledge)
knowledge = knowledge_base["restaurant"][str(19214)]["reviews"][str(1)]["sentences"][str(5)]
print(knowledge)

aspect = knowledge_base["restaurant"][str(19214)]["reviews"][str(2)]["aspect_sentiment"][3]
print(aspect)
aspect = knowledge_base["restaurant"][str(19214)]["reviews"][str(0)]["aspect_sentiment"][7]
print(aspect)
aspect = knowledge_base["restaurant"][str(19214)]["reviews"][str(3)]["aspect_sentiment"][4]
print(aspect)
aspect = knowledge_base["restaurant"][str(19214)]["reviews"][str(1)]["aspect_sentiment"][5]
print(aspect)
'''
entity = knowledge_base["hotel"][str(2)]["name"]
print(entity)

knowledge = knowledge_base["hotel"][str(15)]["reviews"][str(0)]["sentences"][str(2)]
print(knowledge)
'''

knowledge = knowledge_base["restaurant"][str(19214)]["faqs"][str(5)]
print(knowledge)
