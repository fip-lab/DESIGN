import openai
import time
import logging
import argparse
import json



parser = argparse.ArgumentParser(description='Your script description')
parser.add_argument('--response_file', type=str, help='Label file name')
parser.add_argument('--out_file', type=str, help='Output file name')
args = parser.parse_args()


api_key = 'sk-MBHdXOqzaZNZPHTK5462Ac4e8f854e8f882dA87740Bd0a79'
client = openai.OpenAI(api_key = api_key,
                        base_url = "https://www.jcapikey.com/v1")

def get_gpt_response(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        result = response.choices[0].message.content
    except Exception as e:
        logging.info(e)
        result = "error"
        time.sleep(60)

    return result


def create_input(query, knowledge, response_1):
    input = f'''
    Context: {query}
    Knowledge:
    {knowledge}
    Responses:
    {response_1}

    For the given context and knowledge (review/faq), please assess the appropriateness, correctness, and emotional consistency of the response on a scale of 1-5, respectively.
    The assessment indicators are explained as follows:
    -Appropriateness: The response is appropriate given the preceding dialogue.
    -Correctness:The response is able to respond correctly to ideas in knowledge.
    -emotional consistency：The response accurately reflect the tendency to distribute emotions about key aspects of knowledge.
    All you need to provide is the following format:
    Appropriateness: [score]
    Correctness: [score]
    Emotional consistency: [score]
    
    '''
    return input


def loadDate(dataFile):
    dataList =[]
    for line in open(dataFile, "r"):
        dataList.append(json.loads(line))
    return dataList

def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]
dialogue_file = '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/val_data.jsonl'
dialogue_data = loadDate(dialogue_file)
response_data = loadDate(args.response_file)


output_list = []
for i in range(len(dialogue_data)):
    knowledge = """User1's movie preferences:
{movie_similarUser1}
User2's movie preferences:
{movie_similarUser2}""".format(movie_similarUser1 = dialogue_data[i]['userRoleMoviePerfenrnce'], movie_similarUser2 = dialogue_data[i]['systemRoleMoviePerfenrnce'])
    response = response_data[i]['response']
    context = dialogue_data[i]['context']
    dialog = ""
    for i in range(len(context)):
        #按照i来区分user1和user2，并且换行
        if i % 2 == 0:
            dialog += "User1: " + context[i]['text'] + "\n"
        else:
            dialog += "User2: " + context[i]['text'] + "\n"
    #去掉最后一个换行符
    dialog = dialog[:-1]

    input = create_input(dialog, knowledge, response)
    result = get_gpt_response(input)
    print(result)
    output_list.append(result)

with open(args.out_file, "w") as f:
    for output in output_list:
        f.write(json.dumps(output))
        f.write("\n")