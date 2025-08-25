from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



class ABSA:
    def __init__(self, model_name):
        self.ate_tokenizer = AutoTokenizer.from_pretrained("PLM/ate_tk-instruct-base-def-pos-neg-neut-combined")
        self.ate_model = AutoModelForSeq2SeqLM.from_pretrained("PLM/ate_tk-instruct-base-def-pos-neg-neut-combined")
        self.atsc_tokenizer = AutoTokenizer.from_pretrained("PLM/atsc_tk-instruct-base-def-pos-neg-neut-combined")
        self.atsc_model = AutoModelForSeq2SeqLM.from_pretrained("PLM/atsc_tk-instruct-base-def-pos-neg-neut-combined")

        self.ate_instruct = """Definition: The output will be the aspects (both implicit and explicit) which have an associated opinion that are extracted from the input text. In cases where there are no aspects the output should be noaspectterm.
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
        self.atsc_instruct ="""Definition: The output will be 'positive' if the aspect identified in the sentence contains a positive sentiment. If the sentiment of the identified aspect in the input is negative the answer will be 'negative'. 
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

    def get_aspect_term(self, text):
        delim_instruct = ''
        eos_instruct = ' \noutput:'
        tokenized_text = self.ate_tokenizertokenizer(self.ate_instruct + text + delim_instruct + eos_instruct, return_tensors="pt")
        output = self.ate_model.generate(tokenized_text.input_ids)
        result = self.ate_tokenizer.decode(output[0], skip_special_tokens=True)
        if result == 'noaspectterm':
            return ""
        else:
            return result
        
    def get_aspect_sentiment(self, text):
        delim_instruct = ' The aspect is '
        eos_instruct = '.\noutput:'
        aspect_term = self.get_aspect_term(text)
        ##用列表保存多个aspect
        items = aspect_term.split(',')
        # 去除每个项目的前后空格
        aspect_term = [item.strip() for item in items]
            
        ##用字典保存方面和情感
        aspect_sentiment = {}
        for i in aspect_term:
            tokenized_text = self.atsc_tokenizer(self.atsc_instruct + text + delim_instruct + i + eos_instruct, return_tensors="pt")
            output = self.atsc_model.generate(tokenized_text.input_ids)
            sentiment = self.atsc_tokenizer.decode(output[0], skip_special_tokens=True)
            if sentiment == 'positive' or sentiment == 'negative' or sentiment == 'neutral':
                aspect_sentiment[i] = sentiment
        return aspect_sentiment



