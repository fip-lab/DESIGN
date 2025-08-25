import os
import random
import logging
import json
from collections import defaultdict
from itertools import chain
import difflib

import torch
from tqdm import tqdm

from .utils.data import (
    pad_ids, truncate_sequences
)
##jt
from scripts.dataset_walker_2 import DatasetWalker
from scripts.knowledge_reader import KnowledgeReader
from scripts.absa import ABSA

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = {
    "additional_special_tokens": ["<speaker1>", "<speaker2>", "<knowledge_sep>", "<knowledge_tag>","<faq_tag>", "<review_tag>","<user_query>","<absa_of_query>"],
}

with open("data/vocab_dish_or_drink.json", 'r') as f:
    dish_or_drink_vocab = json.load(f)
with open("data/knowledge.json", 'r') as f:
    knowledge_base = json.load(f)
with open("/home/zhuangjt/zhuangjt_disk3/SK-TOD/data/vocab_dish_or_drink.json",'r') as f:
    vocab_dish_or_drink = json.load(f)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.bos = self.tokenizer.bos_token_id
        self.eos = self.tokenizer.eos_token_id
        self.pad = self.tokenizer.pad_token_id
        self.SPECIAL_TOKENS = SPECIAL_TOKENS

        self.speaker1, self.speaker2, self.knowledge_sep, self.knowledge_tag, self.faq_tag, self.review_tag, self.user_query, self.absa = self.tokenizer.convert_tokens_to_ids(
            self.SPECIAL_TOKENS["additional_special_tokens"]
        )
        self.knowledge_sep_token = self.SPECIAL_TOKENS["additional_special_tokens"][2]

        self.faq_prompt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("Whether the faq matches the user query and its aspect items. 1 is a match, 0 is not a match"))
        self.review_prompt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("Whether the review matches the user query and its aspect items. 1 is a match, 0 is not a match"))
        self.both_knowledge_prompt = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("Whether the faq or review matches the user query and its absa infornation. 1 is a match, 0 is not a match"))
        self.dataset_walker = DatasetWalker(split_type, labels=labels, dataroot=self.dataroot, labels_file=labels_file)
        self.dialogs = self._prepare_conversations()
        self.knowledge_reader = KnowledgeReader(self.dataroot, args.knowledge_file)
        self.snippets = self._prepare_knowledge()   ##snippets = {key: {'aspect': aspect_str, 'sentiment': sentiment_str, 'token_ids': tokenized_knowledge}}    aspect:如果多个实体，实体之间逗号连接，如果没有实体则是空格   sentiment类似，但是，如果没有实体，也会生成一个对于整个句子的sentiment
        self._create_examples()

        

        
 
    def _prepare_conversations(self):
        """ Tokenize and encode the dialog data """
        logger.info("Tokenize and encode the dialog data")
        tokenized_dialogs = []
        ##JT
        for i, (log, label, logs_absa,triple) in enumerate(tqdm(self.dataset_walker, disable=False, desc='tokenizing...')):
            dialog = {}
            dialog["id"] = i
            dialog["log"] = log
            if label is not None:
                if "response" in label:
                    label["response_tokenized"] = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(label["response"])
                    )
            dialog["label"] = label
            dialog["log_absa"] = logs_absa
            tokenized_dialogs.append(dialog)
        return tokenized_dialogs

    def _prepare_knowledge(self):
        """ Tokenize and encode the knowledge snippets """
        self.knowledge_docs = self._get_snippet_list()

        tokenized_snippets = defaultdict(dict)
        for snippet_id, snippet in tqdm(enumerate(self.knowledge_docs), disable=False, desc='_prepare_knowledge '):
            key = "{}__{}__{}".format(snippet["domain"], str(snippet["entity_id"]) or "", snippet["doc_id"])
            ##jt
            #knowledge_with_entity_name, knowledge_without_entity_name = self._knowledge_to_string(snippet["doc"], name=snippet["entity_name"] or "")

            ##jt 12-13
            knowledge_with_entity_name, knowledge_without_entity_name = self._knowledge_to_string(snippet["doc"], snippet["absa"], name=snippet["entity_name"] or "")
            #print(knowledge_with_entity_name)
            #print("------------")

            aspect_sentiment = snippet["absa"]
            #print(aspect_sentiment)
            #print(aspeect_sentiment.keys())
            aspects = ", ".join(aspect_sentiment.keys())
            sentiments = ", ".join(aspect_sentiment.values())
            tokenized_snippets[key]['aspect'] = aspects
            tokenized_snippets[key]['sentiment'] = sentiments
            #print("sentence:",knowledge_without_entity_name)
            #print(aspects)
            #rint(sentiments)
            tokenized_knowledge = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(knowledge_with_entity_name))
            tokenized_snippets[key]['token_ids'] = tokenized_knowledge[:self.args.knowledge_max_tokens]
        return tokenized_snippets

    def _get_snippet_list(self):
        """ Get all knowledge snippets in the dataset """
        result = []
        for domain in self.knowledge_reader.get_domain_list():
            for entity_id in self.knowledge_reader.knowledge[domain].keys():
                for review_doc_id in self.knowledge_reader.get_review_doc_ids(domain, entity_id):
                    review_doc = self.knowledge_reader.get_review_doc(domain, entity_id, review_doc_id)
                    for review_sent_id, review_sent in review_doc['sentences'].items():
                        review_absa = self.knowledge_reader.get_review_absa(domain, entity_id, review_doc_id, review_sent_id)
                        result.append(
                            {'domain': domain, 'entity_id': entity_id, 'entity_name': review_doc['entity_name'],
                             'doc_id': f"{review_doc_id}-{review_sent_id}",
                             'doc': {'body': review_sent},
                             'absa': review_absa})
                for faq_doc_id in self.knowledge_reader.get_faq_doc_ids(domain, entity_id):
                    faq_doc = self.knowledge_reader.get_faq_doc(domain, entity_id, faq_doc_id)
                    faq_absa = self.knowledge_reader.get_faq_absa(domain, entity_id, faq_doc_id)
                    #print("faq_absa:",faq_absa)
                    result.append({'domain': domain, 'entity_id': entity_id, 'entity_name': faq_doc['entity_name'],
                                   'doc_id': faq_doc_id,
                                   'doc': {'body': f"{faq_doc['question']} {faq_doc['answer']}"},
                                   'absa': faq_absa})
        return result

    ##JT 12-12
    '''
    def _knowledge_to_string(self, doc, name=""):
        """ Convert a knowledge snippet to a string """
        
        doc_body = f"{name.title()}: {doc['body']}"
        doc_body_2 = doc['body']
    
        return doc_body, doc_body_2
    '''
    def _knowledge_to_string(self, doc, knowledge_absa, name=""):
        knowledge_sentence = doc["body"]
        if len(knowledge_absa) == 1 and "" in knowledge_absa:
            return f"{name.title()}: {knowledge_sentence} [{knowledge_absa['']}]"
        else:
            # Formatting the aspects and their sentiments
            aspects = "; ".join([f"{aspect}:{sentiment}" for aspect, sentiment in knowledge_absa.items()])
            return f"{name.title()}:  {knowledge_sentence} [{aspects}]"

    def _specific_dish_or_drink(self, user_query, vocab_list):
        is_specific_dish_or_drink = False
        matched_entities = []
        for word in vocab_list:
            # 直接匹配
            if word.lower() in user_query.lower() and word not in matched_entities:
                matched_entities.append(word)
        if matched_entities:
            is_specific_dish_or_drink = True
        return is_specific_dish_or_drink, matched_entities
        
        



    def _create_examples(self):
        """ Creating examples for model training and evaluation """
        logger.info("Creating examples")
        self.examples = []
        for dialog in tqdm(self.dialogs, disable=False, desc='creating examples'):
            if self.args.debug > 0 and len(self.examples) >= self.args.debug:
                break
            dialog_id = dialog["id"]
            label = dialog["label"]
            query_absa = dialog["log_absa"]

            dialog = dialog["log"]

            ###JT
            user_last_utterance = dialog[-1]["text"]

            


            if label is None:
                # This will only happen when running knowledge-seeking turn detection on test data
                # So we create dummy target here
                label = {"target": False}

            target = label["target"]

            if not target and self.args.task != "detection":
                # we only care about non-knowledge-seeking turns in turn detection task
                continue

            history = [
                self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn["text"]))
                for turn in dialog
            ]
            gt_resp = label.get("response", "")
            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp))

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            truncated_history = history[-self.args.history_max_utterances:]

            # perform token-level truncation of history from the left 
            truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens)

            if target:
                knowledge_keys = []
                knowledge_candidates = defaultdict(lambda: 0)
                used_knowledge = []
                knowledge_prefix_visited = set()

                ##JT
                knowledge_keys_review = []
                knowledge_keys_faq = []
                used_knowledge_review = []
                used_knowledge_faq = []
                knowledge_candidates_reviews = defaultdict(lambda: 0)
                knowledge_candidates_faqs = defaultdict(lambda: 0)
                knowledge_aspects = []
                knowledge_sentiments = []

                if "knowledge" not in label:
                    raise ValueError("Please run entity matching before running knowledge selection")

                label_knowledge = label["knowledge"]
                
                ##JT
                count_review = 0
                count_faq = 0

                for knowledge in label_knowledge:
                    if not (self.args.task == 'selection' and self.args.eval_only):
                        if knowledge['doc_type'] == 'review':
                            ##JT
                            knowledge_key = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}-{knowledge['sent_id']}"
                            knowledge_key_review = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}-{knowledge['sent_id']}"
                            count_review += 1
                            
                        else:
                            ##JT
                            knowledge_key = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}"
                            knowledge_key_faq = f"{knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}"
                            count_faq += 1

                    # find snippets with same entity as candidates
                    prefix = "{}__{}".format(knowledge["domain"], knowledge["entity_id"])
                    if prefix not in knowledge_prefix_visited:
                        knowledge_prefix_visited.add(prefix)
                        if knowledge["domain"] == "restaurant":
                            entity_name = self.knowledge_reader.get_entity_name(knowledge["domain"], knowledge["entity_id"])
                            vocab_list = vocab_dish_or_drink[entity_name]
                            is_specific_dish_or_drink, matched_entities = self._specific_dish_or_drink(user_last_utterance, vocab_list)
                        
                            if is_specific_dish_or_drink:
                                for review_doc_id in self.knowledge_reader.get_review_doc_ids(knowledge["domain"], knowledge["entity_id"]):
                                    review_doc = self.knowledge_reader.get_review_doc(knowledge["domain"], knowledge["entity_id"], review_doc_id)
                                    for match_entity in matched_entities:
                                        if (match_entity in review_doc['drinks']) or (match_entity in review_doc['dishes']):
                                            _knowledge_candidates = []
                                            for review_sent_id, review_sent in review_doc['sentences'].items():
                                                _knowledge_candidates.append(f"{knowledge['domain']}__{knowledge['entity_id']}__{review_doc_id}-{review_sent_id}")
                                        else:
                                            _knowledge_candidates = [
                                                cand
                                                for cand in self.snippets.keys()
                                                if "__".join(cand.split("__")[:-1]) == prefix
                                            ]
                            else:
                                _knowledge_candidates = [
                                    cand
                                    for cand in self.snippets.keys()
                                    if "__".join(cand.split("__")[:-1]) == prefix
                                ]
                        
                        else:
                            _knowledge_candidates = [
                                cand
                                for cand in self.snippets.keys()
                                if "__".join(cand.split("__")[:-1]) == prefix
                            ]
                        

                        '''
                        _knowledge_candidates = [
                            cand
                            for cand in self.snippets.keys()
                            if "__".join(cand.split("__")[:-1]) == prefix
                        ]
                        '''
                            #print("knowledge:",_knowledge_candidates)


                        for _knowledge_cand_idx, _knowledge_cand in enumerate(_knowledge_candidates):
                            ##JT
                            ##如果_knowledge_cand是 {knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}-{knowledge['sent_id']}的形式，则knowledge_candidates_reviews[_knowledge_cand] = 1
                            ##如果_knowledge_cand是 {knowledge['domain']}__{knowledge['entity_id']}__{knowledge['doc_id']}的形式，则knowledge_candidates_faqs[_knowledge_cand] = 1
                            #判断_knowledge_cands是否存在-
                            #print(_knowledge_cand)
                            #print(_knowledge_cand)
                            knowledge_candidates[_knowledge_cand] = 1

                            if '-' in _knowledge_cand:
                                knowledge_candidates_reviews[_knowledge_cand] = 1
                            else:
                                knowledge_candidates_faqs[_knowledge_cand] = 1
                            
                            
                    if self.split_type == "train" and self.args.negative_sample_method == "oracle":
                        # if there's not enough candidates during training, we just skip this example
                        if len(knowledge_candidates_reviews) + len(knowledge_candidates_faqs) < self.args.n_candidates or len(knowledge_candidates_reviews) + len(knowledge_candidates_faqs) <= len(
                                label["knowledge"]):
                            logger.info("Not enough candidates. Skip this example...")
                            continue

                    if not (self.args.task == 'selection' and self.args.eval_only):
                        used_knowledge.append(
                            self.snippets[knowledge_key]['token_ids'][:self.args.knowledge_max_tokens])
                        knowledge_keys.append(knowledge_key)
                        ##JT
                        if knowledge['doc_type'] == 'review':
                            used_knowledge_review.append(
                                self.snippets[knowledge_key_review]['token_ids'][:self.args.knowledge_max_tokens])
                            knowledge_keys_review.append(knowledge_key_review)
                        elif knowledge['doc_type'] == 'faq':
                            used_knowledge_faq.append(
                                self.snippets[knowledge_key_faq]['token_ids'][:self.args.knowledge_max_tokens])
                            knowledge_keys_faq.append(knowledge_key_faq)
                knowledge_candidates_reviews = [k for k, v in knowledge_candidates_reviews.items()]
                knowledge_candidates_faqs = [k for k, v in knowledge_candidates_faqs.items()]
                knowledge_candidates = [k for k, v in knowledge_candidates.items()]
                
            else:
                
                #knowledge_candidates = None
                used_knowledge = []
                knowledge_keys = []
                
                ##JT
                knowledge_candidates_reviews = None
                knowledge_candidates_faqs = None
                knowledge_candidates = None
                used_knowledge_review = []
                used_knowledge_faq = []
                knowledge_keys_review = []
                knowledge_keys_faq = []




            self.examples.append({
                "history": truncated_history,
                ##JT
                "query_absa": query_absa,  ##aspect term  (string)
                "knowledge_review": used_knowledge_review,      ##转成token_ids的label knowledge
                "knowledge_keys_review": knowledge_keys_review,     ##label knowledge的key
                "candidates_reviews": knowledge_candidates_reviews,     ##候选负样本
                "knowledge_faq": used_knowledge_faq,
                "knowledge_keys_faq": knowledge_keys_faq,
                "candidates_faqs": knowledge_candidates_faqs,
                "knowledge": used_knowledge,
                "knowledge_keys": knowledge_keys,
                "candidates": knowledge_candidates,
                "response": tokenized_gt_resp,
                "response_text": gt_resp,
                "label": label,
                "knowledge_seeking": target,
                "dialog_id": dialog_id
            })

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)


class KnowledgeTurnDetectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeTurnDetectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def build_input_from_segments(self, history):
        """ Build a sequence of input from history """
        instance = {}

        sequence = [[self.cls]] + history[:-1] + [history[-1]]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence0 = [sequence[0]] + sequence_with_speaker[:-1] + [[self.sep]]
        sequence0 = list(chain(*sequence0))
        sequence1 = sequence_with_speaker[-1]
        
        instance["input_ids"] = sequence0 + sequence1
        instance["token_type_ids"] = [0 for _ in sequence0] + [1 for _ in sequence1]
        return instance, sequence

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(example["history"])
        instance["label"] = example["knowledge_seeking"]
        instance["dialog_id"] = example["dialog_id"]
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        labels = [ins["label"] for ins in batch]
        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch]
        }

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        attention_mask = 1 - (input_ids == self.pad).int()
        labels = torch.tensor(labels).long()

        return input_ids, token_type_ids, attention_mask, labels, data_info


class KnowledgeSelectionDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(KnowledgeSelectionDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

        if self.args.negative_sample_method not in ["all", "mix", "oracle"]:
            # Negative sampling method for knowledge selection
            # all: use all knowledge snippets of all entities as candidates
            # oracle: use all knowledge snippets of oracle entities as candidates
            # mix: use oracle candidates & equally sized candidates sampled from other entities
            raise ValueError(
                "negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)
    '''
    def _knowledge_to_string(self, doc, name=""):
        """ convert a knowlege snippet to a string """
        join_str = " %s " % self.knowledge_sep_token
        doc_body = doc['body']
        knowledge_string = join_str.join([name.title(), doc_body])
        return knowledge_string, doc_body
    '''
    def _knowledge_to_string(self, doc, knowledge_absa, name=""):
        knowledge_sentence = doc["body"]
        join_str = " %s " % self.knowledge_sep_token
        if len(knowledge_absa) == 1 and "" in knowledge_absa:
            doc_body =  f"{knowledge_sentence} [{knowledge_absa['']}]"
        else:
            # Formatting the aspects and their sentiments
            aspects = "; ".join([f"{aspect}:{sentiment}" for aspect, sentiment in knowledge_absa.items()])
            doc_body = f"{name.title()}:  {knowledge_sentence} [{aspects}]"
        knowledge_string = join_str.join([name.title(), doc_body])
        return knowledge_string, doc_body


    def __getitem__(self, index):
        example = self.examples[index]

        this_inst = {
            "dialog_id": example["dialog_id"],
            "input_ids": [],
            "token_type_ids": []
        }

        

        if self.split_type != "train":
            # if eval_all_snippets is set, we use all snippets as candidates with no sampling
            if self.args.eval_all_snippets:
                candidates = list(self.snippets.keys())
            else:
                ##jt
                if self.args.selection_knowledge_type == 'review':
                    candidates = example["candidates_reviews"]
                elif self.args.selection_knowledge_type == 'faq':
                    candidates = example["candidates_faqs"]
                elif self.args.selection_knowledge_type == 'both':
                    candidates = example["candidates"]
                else:
                    raise ValueError(
                        "selection_knowledge_type must be review,faq or both, got %s" % self.args.selection_knowledge_type)

        else:
            if self.args.negative_sample_method == "all":
                candidates = list(self.snippets.keys())
            elif self.args.negative_sample_method == "mix":
                candidates = example["candidates"] + random.sample(list(self.snippets.keys()),
                                                                   k=len(example["candidates"]))
            elif self.args.negative_sample_method == "oracle":
                ##jt
                if self.args.selection_knowledge_type == 'review':
                    #print("进入了review 的KnowledgeSelectionDataset")
                    candidates = example["candidates_reviews"]

                elif self.args.selection_knowledge_type == 'faq':
                    #print("进入了faq的KnowledgeSelectionDataset")
                    candidates = example["candidates_faqs"]
                elif self.args.selection_knowledge_type == 'both':
                    #print("进入了both的KnowledgeSelectionDataset")
                    candidates = example["candidates"]
                else:
                    raise ValueError(
                        "selection_knowledge_type must be review,faq or both, got %s" % self.args.selection_knowledge_type)
            else:  # although we have already checked for this, still adding this here to be sure
                raise ValueError(
                    "negative_sample_method must be all, mix, or oracle, got %s" % self.args.negative_sample_method)

        candidate_keys = candidates
        this_inst["candidate_keys"] = candidate_keys
        candidates = [self.snippets[cand_key]['token_ids'] for cand_key in candidates]

        if self.split_type == "train":
            if self.args.selection_knowledge_type == 'review':
                candidates = self._shrink_label_cands(example["knowledge_review"], candidates)
                label_idx = [candidates.index(knowledge) for knowledge in example["knowledge_review"]]
            elif self.args.selection_knowledge_type == 'faq':
                candidates = self._shrink_label_cands(example["knowledge_faq"], candidates)
                label_idx = [candidates.index(knowledge) for knowledge in example["knowledge_faq"]]
            elif self.args.selection_knowledge_type == 'both':
                candidates = self._shrink_label_cands(example["knowledge"], candidates)
                label_idx = [candidates.index(knowledge) for knowledge in example["knowledge"]]
            else:
                raise ValueError(
                    "selection_knowledge_type must be review,faq or both, got %s" % self.args.selection_knowledge_type)
        else:
            if self.args.selection_knowledge_type == 'review':
                label_idx = [candidates.index(knowledge) for knowledge in example["knowledge_review"]]
            elif self.args.selection_knowledge_type == 'faq':
                label_idx = [candidates.index(knowledge) for knowledge in example["knowledge_faq"]]
            elif self.args.selection_knowledge_type == 'both':
                label_idx = [candidates.index(knowledge) for knowledge in example["knowledge"]]

        this_inst["label_idx"] = label_idx
        for cand in candidates:
            instance = self.build_input_from_segments(
                cand,
                example["history"],
                example["query_absa"]
            )
            this_inst["input_ids"].append(instance["input_ids"])
            this_inst["token_type_ids"].append(instance["token_type_ids"])

        return this_inst

    def build_input_from_segments(self, knowledge, history, query_absa):
        """ Build a sequence of input from 2 segments: knowledge and history"""
        instance = {}
        '''
        sequence = [[self.cls]] + history
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]
        sequence_with_speaker = list(chain(*sequence_with_speaker))

        sequence0 = [self.cls] + sequence_with_speaker + [self.sep]
        sequence1 = knowledge + [self.sep]

        if 'roberta' in str(type(self.tokenizer)):
            sequence0 += [self.sep]
        instance["input_ids"] = sequence0 + sequence1
        instance["token_type_ids"] = [0 for _ in sequence0] + [1 for _ in sequence1]
        return instance, sequence
        '''


        #JT
        #aspect_sentence = "The aspect item of query is " + aspect_item + "."
        try:
            if query_absa["aspect"]=="" and query_absa["sentiment"] !={}:
                absa_sentence = "The emotional polarity of query is " + query_absa["sentiment"][""] + "."
            elif query_absa["aspect"]=="" and query_absa["sentiment"] =={}:
                absa_sentence = ""
            else:
                if query_absa["sentiment"] =={}:
                    absa_sentence = ""
                else:
                # Extract aspects and sentiments
                    aspects = query_absa["aspect"].split(", ")
                    sentiments = []
                    sentiments.extend(query_absa["sentiment"].values())

                    #sentiments = [query_absa["sentiment"][aspect] for aspect in aspects]
                    # Check the number of aspects to determine the use of 'is' or 'are'
                    verb = "is" if len(aspects) == 1 else "are"
                    # Construct the sentence
                    if verb == "is":
                        absa_sentence = f"The aspect item of query {verb} {', '.join(aspects)}. And the emotional polarity of it is {', '.join(sentiments)}"
                    else:
                        absa_sentence = f"The aspect item of query {verb} {', '.join(aspects)}. And the emotional polarity of them is {', '.join(sentiments)}"
        except:
            print("query_absa:")
            print(query_absa)
        #print(absa_sentence)
        #print("------------")
        absa_sentence_tokenized = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(absa_sentence))


        if self.args.selection_knowledge_type == 'review':
            sequence0 = [self.cls] + [self.user_query] + history[-1] + [self.sep] + [self.absa] + absa_sentence_tokenized + [self.sep] + [self.review_tag] + knowledge + [self.sep]
            sequence1 = self.review_prompt + [self.sep]
        elif self.args.selection_knowledge_type == 'faq':
            sequence0 = [self.cls] + [self.user_query] + history[-1] + [self.sep] + [self.absa] + absa_sentence_tokenized + [self.sep] + [self.faq_tag] + knowledge + [self.sep]
            sequence1 = self.faq_prompt + [self.sep]
        elif self.args.selection_knowledge_type == 'both':
            sequence0 = [self.cls] + [self.user_query] + history[-1] + [self.sep] + [self.absa] + absa_sentence_tokenized + [self.sep] + [self.review_tag] + knowledge + [self.sep]
            sequence1 = self.both_knowledge_prompt + [self.sep]
        instance["input_ids"] = sequence0 + sequence1
        instance["token_type_ids"] = [0 for _ in sequence0] + [1 for _ in sequence1]

        return instance



    def _shrink_label_cands(self, label, candidates):
        """ remove positive knowledge snippets from the candidates """
        shrunk_label_cands = candidates.copy()
        for l in label:
            if l in shrunk_label_cands:
                shrunk_label_cands.remove(l)

        ##JT
        if self.args.selection_knowledge_type == 'review':
            sample_size = min(len(label), len(shrunk_label_cands))
        elif self.args.selection_knowledge_type == 'faq':
            if len(label) < 2:
                sample_size = 2
            else:
                sample_size = min(len(label), len(shrunk_label_cands))
        elif self.args.selection_knowledge_type == 'both':
            sample_size = min(len(label), len(shrunk_label_cands))
        shrunk_label_cands = random.sample(shrunk_label_cands, k=sample_size)

        shrunk_label_cands.extend(label)
        random.shuffle(shrunk_label_cands)
        return shrunk_label_cands

    def collate_fn(self, batch):
        input_ids = [ids for ins in batch for ids in ins["input_ids"]]
        token_type_ids = [ids for ins in batch for ids in ins["token_type_ids"]]
        label_idx = [1 if i in ins['label_idx'] else 0 for ins in batch for i in range(len(ins['input_ids']))]
        data_info = {
            "dialog_ids": [ins["dialog_id"] for ins in batch],
            "candidate_keys": [ins["candidate_keys"] for ins in batch]
        }

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        attention_mask = 1 - (input_ids == self.pad).int()
        token_type_ids = torch.tensor(pad_ids(token_type_ids, 0))
        label_idx = torch.tensor(label_idx)
        return input_ids, token_type_ids, attention_mask, label_idx, data_info


class ResponseGenerationDataset(BaseDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(
            example["knowledge"],
            example["history"],
            example["response"]
        )
        return instance

    def build_input_from_segments(self, knowledge, history, response):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}
        knowledge = [[self.knowledge_sep] + k for k in knowledge]
        knowledge = [w for k in knowledge for w in k]

        # 3: special tokens; len(history): special speaker tokens
        entire_input_len = self.tokenizer.model_max_length - 3

        entire_knowledge_len, entire_history_len = len(knowledge), len(list(chain(*history)))
        max_history_len = int((entire_history_len * entire_input_len) / (entire_knowledge_len + entire_history_len))
        max_history_len = min(entire_history_len + len(history), max(max_history_len, 256))
        max_knowledge_len = entire_input_len - max_history_len  # - len(history)

        if max_knowledge_len < entire_knowledge_len:
            logger.warning(
                f"Knowledge too long! Have been truncated from {entire_knowledge_len} to {max_knowledge_len}")
            knowledge = knowledge[:max_knowledge_len]
        if max_history_len < entire_history_len:
            logger.warning(f"History too long! Have been truncated from {entire_history_len} to {max_history_len}")

        sequence = [knowledge] + history + [response]
        sequence_with_speaker = [
            [self.speaker1 if (len(sequence) - i) % 2 == 0 else self.speaker2] + s
            for i, s in enumerate(sequence[1:])
        ]  # speaker 2 (user)
        history = list(chain(*sequence_with_speaker[:-1]))[:max_history_len]
        sequence = [[self.bos]] + [sequence[0]] + [[self.knowledge_tag]] + [history] + [[self.eos]]
        instance["input_ids"] = list(chain(*sequence))
        instance["lm_labels"] = [self.bos] + sequence_with_speaker[-1] + [self.eos]
        return instance, sequence

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        attention_mask = 1 - (input_ids == self.pad).int()
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))

        return input_ids, attention_mask, lm_labels


class ResponseGenerationEvalDataset(ResponseGenerationDataset):
    def __init__(self, args, tokenizer, split_type, labels=True, labels_file=None):
        super(ResponseGenerationEvalDataset, self).__init__(args, tokenizer, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch
