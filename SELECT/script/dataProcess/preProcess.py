import json

def loadDate(dataFile):
    dataList =[]
    for line in open(dataFile, "r"):
        dataList.append(json.loads(line))
    return dataList


def savaDate(dataList, dataFile):
    with open(dataFile, "w") as f:
        for data in dataList:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            

def remove_movieYear(title):
    try:
        return title.split(" (")[0]
    except:
        return title


#合并同一角色连续对话
def mergeSameRoleDialogues(message):
    result = []
    current = None

    for entry in message:
        if current and entry['senderWorkerId'] == current['senderWorkerId']:
            current['text'] += ' ' + entry['text']
        else:
            if current:
                result.append({'senderWorkerId': current['senderWorkerId'], 'text': current['text']})
            current = {'senderWorkerId': entry['senderWorkerId'], 'text': entry['text']}

    # Append the last accumulated entry
    if current:
        result.append({'senderWorkerId': current['senderWorkerId'], 'text': current['text']})
    return result


#对moviePerfenrnce做处理
def processMoviePerfenrnce(moviePerfenrnce, movieDict):  
    newDict = {}
    for k, v in moviePerfenrnce.items():
        if v['liked'] > 0:
            emotion = 'recommend'
        else:
            emotion = 'neutral' 
        #key是一个电影的id，现在我希望将其替换为电影的名字
        movie_name = movieDict[int(k)]
        newDict[movie_name] = emotion
         
    return newDict

    



#############################################################################################################

dataFile = '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Select/candidate_data.jsonl'
dataList = loadDate(dataFile)
print(len(dataList))

newDataList = []

for data in dataList:
    movieDict = data['movieMentions']
    #如果movieDict中有value是None,就将这个data删除
    if type(movieDict) == list or None in movieDict.values() :
        dataList.remove(data)
        continue
    movieDict = {int(k): remove_movieYear(v) for k, v in movieDict.items()}

    message = data['messages']
    roleOne = data['respondentWorkerId']
    roleTwo = data['initiatorWorkerId']
    roleOneMoviePerfenrnce = data['respondentQuestions']
    roleTwoMoviePerfenrnce = data['initiatorQuestions']

    if(type(roleOneMoviePerfenrnce) == list or type(roleTwoMoviePerfenrnce) == list):
        dataList.remove(data)
        continue

    #先对对话做处理
    message = mergeSameRoleDialogues(message)
    #去掉最后一轮的两次对话和评论句以及开场白
    message = message[:-3]
    message = message[1:]
    #查看目前的最后一次对话的角色
    #如果message的长度小于2，直接将这个data删除
    if len(message) < 2:
        dataList.remove(data)
        continue

    systemRole = message[-1]['senderWorkerId']
    userRole = roleOne if systemRole == roleTwo else roleTwo
    systemRoleMoviePerfenrnce = roleOneMoviePerfenrnce if systemRole == roleOne else roleTwoMoviePerfenrnce
    userRoleMoviePerfenrnce = roleOneMoviePerfenrnce if systemRole == roleTwo else roleTwoMoviePerfenrnce
    
    #然后从前往后找，找到anotherRole第一次出现的位置，然后将这个位置之前的对话去掉
    for i in range(len(message)):
        if message[i]['senderWorkerId'] == userRole:
            message = message[i:]
            break
    
    #对于message中的每一个句子，如果出现@173785这种形式的，就将其替换为对应的电影名
    for mes in message:
        for movie_id, movie_title in movieDict.items():
            mes['text'] = mes['text'].replace(f"@{movie_id}", movie_title)
    #经过处理之后，message中的第一句就是用来匹配的query,最后一句是reference,最后一句之前的句子都是context


    #然后对MoviePerfenrnce做处理
    systemRoleMoviePerfenrnce = processMoviePerfenrnce(systemRoleMoviePerfenrnce, movieDict)
    userRoleMoviePerfenrnce = processMoviePerfenrnce(userRoleMoviePerfenrnce, movieDict)

    newData = {
        'context': message[:-1],
        'reference': message[-1],
        'systemRoleMoviePerfenrnce': systemRoleMoviePerfenrnce,
        'userRoleMoviePerfenrnce': userRoleMoviePerfenrnce
    }
    newDataList.append(newData)
print(len(newDataList))
#保存处理之后的数据，美观一点，保存为json格式
savaDate(newDataList, '/home/zhuangjt/zhuangjt_disk3/SELECT/data/ReDial/ReDial_Postprocess/candidate_data.jsonl')



