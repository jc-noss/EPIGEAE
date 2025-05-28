import os
import re
import sys
sys.path.append("../")
import torch
import numpy as np

from torch.utils.data import Dataset
from processors.processor_base import DSET_processor
from utils import EXTERNAL_TOKENS, _PREDEFINED_QUERY_TEMPLATE

# 组织和存储事件抽取任务中单个样本的所有相关信息而设计的，包括输入数据的预处理、模型所需的各种ID序列、预测和真实标签的管理等
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, feature_id, 
                event_type, event_trigger,
                enc_text, enc_input_ids, enc_mask_ids, 
                dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
                arg_quries, arg_joint_prompt, target_info,
                old_tok_to_new_tok_index = None, full_text = None, arg_list=None,
        ):
        # 设置基本属性，包括示例ID、特征ID、事件类型和事件触发词。
        self.example_id = example_id
        self.feature_id = feature_id
        self.event_type = event_type
        self.event_trigger = event_trigger
        # 编码文本(enc_text)是经过初步处理的文本字符串；
        # enc_input_ids和enc_mask_ids分别是编码文本转换为模型输入的tokenID序列和对应的注意力掩码，用于指示哪些部分需要模型关注。
        self.enc_text = enc_text
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids
        # 解码提示相关属性，用于特定的提示查询机制。包括解码提示文本、其对应的tokenID序列及掩码。
        self.dec_prompt_texts = dec_prompt_text
        self.dec_prompt_ids =dec_prompt_ids
        self.dec_prompt_mask_ids=dec_prompt_mask_ids


        # 根据是否有论元查询 (arg_quries)，分别设置解码器中论元查询的token ID、掩码、起始和结束位置，以及目标信息中的起始和结束位置ID。
        # 如果没有论元查询，则设置为 None。
        if arg_quries is not None:
            self.dec_arg_query_ids = [v[0] for k,v in arg_quries.items()]
            self.dec_arg_query_masks = [v[1] for k,v in arg_quries.items()]
            self.dec_arg_start_positions = [v[2] for k,v in arg_quries.items()]
            self.dec_arg_end_positions = [v[3] for k,v in arg_quries.items()]
            self.start_position_ids = [v['span_s'] for k,v in target_info.items()]
            self.end_position_ids = [v['span_e'] for k,v in target_info.items()]
        else:
            self.dec_arg_query_ids = None
            self.dec_arg_query_masks = None
        
        self.arg_joint_prompt = arg_joint_prompt
        
        self.target_info = target_info
        self.old_tok_to_new_tok_index = old_tok_to_new_tok_index

        self.full_text = full_text
        self.arg_list = arg_list

    # 初始化预测结果的存储结构，用于保存模型预测的论元角色及其对应的跨度
    def init_pred(self):
        self.pred_dict_tok = dict()
        self.pred_dict_word = dict()
    
    # 向预测结果中添加新的预测跨度，同时支持从token级跨度转换到词级跨度。
    def add_pred(self, role, span, dset_type):
        if role not in self.pred_dict_tok:
            self.pred_dict_tok[role] = list()
        if span not in self.pred_dict_tok[role]:
            self.pred_dict_tok[role].append(span)

            if span!=(0, 0):
                if role not in self.pred_dict_word:
                    self.pred_dict_word[role] = list()
                word_span = self.get_word_span(span, dset_type)         # convert token span to word span 
                if word_span not in self.pred_dict_word[role]:
                    self.pred_dict_word[role].append(word_span)

    # 设置真实标签（Ground Truth, GT），根据模型类型和数据集类型处理目标信息，转换为token级和词级的真实标签跨度。
    def set_gt(self, model_type, dset_type):
        self.gt_dict_tok = dict()
        if model_type == 'base':
            for k,v in self.target_info.items():
                span_s = list(np.where(v["span_s"])[0])
                span_e = list(np.where(v["span_e"])[0])
                self.gt_dict_tok[k] = [(s,e) for (s,e) in zip(span_s, span_e)]
        elif "epig_eae" in model_type:
            for k,v in self.target_info.items():
                self.gt_dict_tok[k] = [(s,e) for (s,e) in zip(v["span_s"], v["span_e"])]
        else:
            assert(0==1)

        self.gt_dict_word = dict()
        for role in self.gt_dict_tok:
            for span in self.gt_dict_tok[role]:
                if span!=(0, 0):
                    if role not in self.gt_dict_word:
                        self.gt_dict_word[role] = list()
                    word_span = self.get_word_span(span, dset_type)
                    self.gt_dict_word[role].append(word_span)

    #  创建一个属性方法，提供从新token索引到旧token索引的映射，方便索引转换。
    @property
    def old_tok_index(self):
        new_tok_index_to_old_tok_index = dict()
        for old_tok_id, (new_tok_id_s, new_tok_id_e) in enumerate(self.old_tok_to_new_tok_index):
            for j in range(new_tok_id_s, new_tok_id_e):
                new_tok_index_to_old_tok_index[j] = old_tok_id 
        return new_tok_index_to_old_tok_index

    # 根据给定的token跨度和数据集类型，计算并返回对应的词跨度
    def get_word_span(self, span, dset_type):
        """
        Given features with gt/pred token-spans, output gt/pred word-spans
        """
        if span==(0, 0):
            raise AssertionError()
        offset = 0 if dset_type=='ace_eeqa' else self.event_trigger[2]
        span = list(span)
        span[0] = min(span[0], max(self.old_tok_index.keys()))
        span[1] = max(span[1]-1, min(self.old_tok_index.keys()))

        while span[0] not in self.old_tok_index:
            span[0] += 1 
        span_s = self.old_tok_index[span[0]] + offset
        while span[1] not in self.old_tok_index:
            span[1] -= 1 
        span_e = self.old_tok_index[span[1]] + offset
        while span_e < span_s:
            span_e += 1
        return (span_s, span_e)

    # 重写 __repr__ 方法，使得当尝试打印或表示 InputFeatures 实例时，会输出一些关键属性的信息，便于调试和理解实例状态。
    def __repr__(self):
        s = "" 
        s += "example_id: {}\n".format(self.example_id)
        s += "event_type: {}\n".format(self.event_type)
        s += "trigger_word: {}\n".format(self.event_trigger)
        s += "old_tok_to_new_tok_index: {}\n".format(self.old_tok_to_new_tok_index)
        
        s += "enc_input_ids: {}\n".format(self.enc_input_ids)
        s += "enc_mask_ids: {}\n".format(self.enc_mask_ids)
        s += "dec_prompt_ids: {}\n".format(self.dec_prompt_ids)
        s += "dec_prompt_mask_ids: {}\n".format(self.dec_prompt_mask_ids)
        return s

# 将特征实例转化为可以迭代的数据集
class ArgumentExtractionDataset(Dataset):
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
 
    @staticmethod
    def collate_fn(batch):
        
        enc_input_ids = torch.tensor([f.enc_input_ids for f in batch])
        enc_mask_ids = torch.tensor([f.enc_mask_ids for f in batch])

        if  batch[0].dec_prompt_ids is not None:
            dec_prompt_ids = torch.tensor([f.dec_prompt_ids for f in batch])
            dec_prompt_mask_ids = torch.tensor([f.dec_prompt_mask_ids for f in batch])
        else:
            dec_prompt_ids=None
            dec_prompt_mask_ids=None

        example_idx = [f.example_id for f in batch]
        feature_idx = torch.tensor([f.feature_id for f in batch])

        if batch[0].dec_arg_query_ids is not None:
            dec_arg_query_ids = [torch.LongTensor(f.dec_arg_query_ids) for f in batch]
            dec_arg_query_mask_ids = [torch.LongTensor(f.dec_arg_query_masks) for f in batch]
            dec_arg_start_positions = [torch.LongTensor(f.dec_arg_start_positions) for f in batch]
            dec_arg_end_positions = [torch.LongTensor(f.dec_arg_end_positions) for f in batch]
            start_position_ids = [torch.FloatTensor(f.start_position_ids) for f in batch]
            end_position_ids = [torch.FloatTensor(f.end_position_ids) for f in batch]
        else:
            dec_arg_query_ids = None
            dec_arg_query_mask_ids = None
            dec_arg_start_positions = None
            dec_arg_end_positions = None
            start_position_ids = None
            end_position_ids = None

        target_info = [f.target_info for f in batch]
        old_tok_to_new_tok_index = [f.old_tok_to_new_tok_index for f in batch]
        arg_joint_prompt = [f.arg_joint_prompt for f in batch]
        arg_lists = [f.arg_list for f in batch ]


        event_trigger = [f.event_trigger for f in batch]
        event_type = [f.event_type for f in batch]
        full_text=[f.full_text for f in batch]

        return enc_input_ids, enc_mask_ids, \
                dec_arg_query_ids, dec_arg_query_mask_ids,\
                dec_prompt_ids, dec_prompt_mask_ids,\
                target_info, old_tok_to_new_tok_index, arg_joint_prompt, arg_lists, \
                example_idx, feature_idx, \
                dec_arg_start_positions, dec_arg_end_positions, \
                start_position_ids, end_position_ids, \
                event_trigger, event_type,full_text#新增


class MultiargProcessor(DSET_processor):
    def __init__(self, args, tokenizer):
        super().__init__(args, tokenizer) 
        self.set_dec_input()
        self.collate_fn = ArgumentExtractionDataset.collate_fn
    def set_dec_input(self):
        self.arg_query=False
        self.prompt_query=False
        if self.args.model_type == "base":
            self.arg_query = True
        elif "epig_eae" in self.args.model_type:
            self.prompt_query = True
        else:
            raise NotImplementedError(f"Unexpected setting {self.args.model_type}")
    @staticmethod
    def _read_prompt_group(prompt_path):
        with open(prompt_path) as f:
            lines = f.readlines()
        prompts = dict()
        for line in lines:
            if not line:
                continue
            event_type, prompt = line.split(":")
            prompts[event_type] = prompt
        return prompts

    def create_dec_qury(self, arg, event_trigger):
        dec_text = _PREDEFINED_QUERY_TEMPLATE.format(arg=arg, trigger=event_trigger)
                
        dec = self.tokenizer(dec_text)
        dec_input_ids, dec_mask_ids = dec["input_ids"], dec["attention_mask"]

        while len(dec_input_ids) < self.args.max_dec_seq_length:
            dec_input_ids.append(self.tokenizer.pad_token_id)
            dec_mask_ids.append(self.args.pad_mask_token)

        matching_result = re.search(arg, dec_text)
        char_idx_s, char_idx_e = matching_result.span(); char_idx_e -= 1
        tok_prompt_s = dec.char_to_token(char_idx_s)
        tok_prompt_e = dec.char_to_token(char_idx_e) + 1
        # tok_prompt_s = char_idx_s
        # tok_prompt_e = char_idx_e + 1

        return dec_input_ids, dec_mask_ids, tok_prompt_s, tok_prompt_e

    def convert_examples_to_features_oeecfc(self, examples):
        if self.prompt_query:
            prompts = self._read_prompt_group(self.args.prompt_path)
        if os.environ.get("DEBUG", False): counter = [0, 0, 0]
        features = []
        for example_idx, example in enumerate(examples):
            example_id = example.doc_id
            sent = example.sent
            event_type = example.type
            event_args = example.args

            trigger_start, trigger_end = example.trigger['start'], example.trigger['end']
            event_trigger = [example.trigger['text'], [trigger_start, trigger_end], example.trigger['offset']]

            event_args_name = [arg['role'] for arg in event_args]
            if os.environ.get("DEBUG", False): counter[2] += len(event_args_name)

            sent = sent[:trigger_start] + ['<t>'] + sent[trigger_start:trigger_end] + ['</t>'] + sent[trigger_end:]
            enc_text = "".join(sent)

            old_tok_to_char_index = []  # old tok: split by oneie
            old_tok_to_new_tok_index = []  # new tok: split by BART

            curr = 0
            for tok in sent:
                if tok not in EXTERNAL_TOKENS:
                    old_tok_to_char_index.append([curr, curr + len(tok)])  # exact word start char and end char index
                curr += len(tok)

            enc = self.tokenizer(enc_text)
            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            if len(enc_input_ids) > self.args.max_enc_seq_length:
                enc_input_ids = enc["input_ids"][:self.args.max_enc_seq_length]
                enc_mask_ids = enc["attention_mask"][:self.args.max_enc_seq_length]
            while len(enc_input_ids) < self.args.max_enc_seq_length:
                enc_input_ids.append(self.tokenizer.pad_token_id)
                enc_mask_ids.append(self.args.pad_mask_token)

            temp_new_tok_e = 0
            for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(old_tok_to_char_index):
                new_tok_s = enc.char_to_token(char_idx_s)
                new_tok_e = enc.char_to_token(char_idx_e)
                if not new_tok_e and not new_tok_s:
                    new_tok_s = temp_new_tok_e
                    new_tok_e = new_tok_s + 1
                elif not new_tok_s:
                    new_tok_s = new_tok_e - 1
                elif not new_tok_e:
                    new_tok_e = new_tok_s + 1
                new_tok = [new_tok_s, new_tok_e]
                old_tok_to_new_tok_index.append(new_tok)

            # Deal with prompt template
            if self.prompt_query:
                dec_prompt_text = prompts[event_type].strip()
                if dec_prompt_text:
                    if event_trigger[0]:
                        dec_prompt_text = dec_prompt_text.replace('trigger', event_trigger[0])
                    else:
                        dec_prompt_text = dec_prompt_text.replace('trigger ', '')
                    dec_prompt = self.tokenizer(dec_prompt_text)
                    dec_prompt_ids, dec_prompt_mask_ids = dec_prompt["input_ids"], dec_prompt["attention_mask"]
                    assert len(dec_prompt_ids) <= self.args.max_prompt_seq_length
                    while len(dec_prompt_ids) < self.args.max_prompt_seq_length:
                        dec_prompt_ids.append(self.tokenizer.pad_token_id)
                        dec_prompt_mask_ids.append(self.args.pad_mask_token)
                else:
                    raise ValueError(f"no prompt provided for event: {event_type}")
            else:
                dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids = None, None, None

            arg_list = self.argument_dict[event_type.replace(':', '.')]
            arg_quries = dict()
            arg_joint_prompt = dict()
            target_info = dict()
            if os.environ.get("DEBUG", False): arg_set = set()
            for arg in arg_list:
                arg_query = None
                prompt_slots = None
                arg_target = {
                    "text": list(),
                    "span_s": list(),
                    "span_e": list()
                }

                if self.arg_query:
                    arg_query = self.create_dec_qury(arg, event_trigger[0])
                if self.prompt_query:
                    prompt_slots = {
                        "tok_s": list(), "tok_e": list(),
                    }

                    for matching_result in re.finditer(r'\b' + re.escape(arg) + r'\b', dec_prompt_text.split('.')[0]):
                        char_idx_s, char_idx_e = matching_result.span();
                        char_idx_e -= 1
                        tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
                        tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
                        prompt_slots["tok_s"].append(tok_prompt_s);
                        prompt_slots["tok_e"].append(tok_prompt_e)
                answer_texts, start_positions, end_positions = list(), list(), list()
                if arg in event_args_name:
                    if os.environ.get("DEBUG", False): arg_set.add(arg)
                    arg_idxs = [i for i, x in enumerate(event_args_name) if x == arg]
                    if os.environ.get("DEBUG", False): counter[0] += 1; counter[1] += len(arg_idxs)

                    for arg_idx in arg_idxs:
                        event_arg_info = event_args[arg_idx]
                        answer_text = event_arg_info['text'];
                        answer_texts.append(answer_text)
                        start_old, end_old = event_arg_info['start'], event_arg_info['end']
                        start_position = old_tok_to_new_tok_index[start_old][0];
                        start_positions.append(start_position)
                        end_position = old_tok_to_new_tok_index[end_old - 1][1];
                        end_positions.append(end_position)

                if self.arg_query:
                    arg_target["span_s"] = [1 if i in start_positions else 0 for i in
                                            range(self.args.max_enc_seq_length)]
                    arg_target["span_e"] = [1 if i in end_positions else 0 for i in range(self.args.max_enc_seq_length)]
                    if sum(arg_target["span_s"]) == 0:
                        arg_target["span_s"][0] = 1
                        arg_target["span_e"][0] = 1
                if self.prompt_query:
                    arg_target["span_s"] = start_positions
                    arg_target["span_e"] = end_positions

                arg_target["text"] = answer_texts
                arg_quries[arg] = arg_query
                arg_joint_prompt[arg] = prompt_slots
                target_info[arg] = arg_target

            if not self.arg_query:
                arg_quries = None
            if not self.prompt_query:
                arg_joint_prompt = None

            feature_idx = len(features)
            features.append(
                InputFeatures(example_id, feature_idx,
                              event_type, event_trigger,
                              enc_text, enc_input_ids, enc_mask_ids,
                              dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
                              arg_quries, arg_joint_prompt, target_info,
                              old_tok_to_new_tok_index=old_tok_to_new_tok_index, full_text=example.full_text,
                              arg_list=arg_list
                              )
            )

        if os.environ.get("DEBUG", False): print(
            '\033[91m' + f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})" + '\033[0m')
        return features
    def convert_examples_to_features(self, examples):
        if self.prompt_query:
            prompts = self._read_prompt_group(self.args.prompt_path)

        if os.environ.get("DEBUG", False): counter = [0, 0, 0]
        features = []
        for example_idx, example in enumerate(examples):
            example_id =example.doc_id
            sent = example.sent  
            event_type = example.type
            event_args = example.args

            trigger_start, trigger_end = example.trigger['start'], example.trigger['end']
            # NOTE: extend trigger full info in features
            event_trigger = [example.trigger['text'], [trigger_start, trigger_end], example.trigger['offset']]

            event_args_name = [arg['role'] for arg in event_args]
            if os.environ.get("DEBUG", False): counter[2] += len(event_args_name)

            sent = sent[:trigger_start] + ['<t>'] + sent[trigger_start:trigger_end] + ['</t>'] + sent[trigger_end:]
            enc_text = " ".join(sent)

            # change the mapping to idx2tuple (start/end word idx)
            old_tok_to_char_index = []     # old tok: split by oneie
            old_tok_to_new_tok_index = []  # new tok: split by BART

            curr = 0
            for tok in sent:
                if tok not in EXTERNAL_TOKENS:
                    old_tok_to_char_index.append([curr, curr+len(tok)-1]) # exact word start char and end char index
                curr += len(tok)+1

            enc = self.tokenizer(enc_text)
            enc_input_ids, enc_mask_ids = enc["input_ids"], enc["attention_mask"]
            if len(enc_input_ids)> self.args.max_enc_seq_length:
                raise ValueError(f"Please increase max_enc_seq_length above {len(enc_input_ids)}")
            while len(enc_input_ids) < self.args.max_enc_seq_length:
                enc_input_ids.append(self.tokenizer.pad_token_id)
                enc_mask_ids.append(self.args.pad_mask_token)

            for old_tok_idx, (char_idx_s, char_idx_e) in enumerate(old_tok_to_char_index):
                new_tok_s = enc.char_to_token(char_idx_s)
                new_tok_e = enc.char_to_token(char_idx_e) + 1
                new_tok = [new_tok_s, new_tok_e]
                old_tok_to_new_tok_index.append(new_tok)

            # Deal with prompt template
            if self.prompt_query:
                dec_prompt_text = prompts[event_type].strip()
                if dec_prompt_text:
                    dec_prompt = self.tokenizer(dec_prompt_text)
                    dec_prompt_ids, dec_prompt_mask_ids = dec_prompt["input_ids"], dec_prompt["attention_mask"]
                    assert len(dec_prompt_ids)<=self.args.max_prompt_seq_length, f"\n{example}\n{arg_list}\n{dec_prompt_text}"
                    while len(dec_prompt_ids) < self.args.max_prompt_seq_length:
                        dec_prompt_ids.append(self.tokenizer.pad_token_id)
                        dec_prompt_mask_ids.append(self.args.pad_mask_token)
                else:
                    raise ValueError(f"no prompt provided for event: {event_type}")
            else:
                dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids = None, None, None

            arg_list = self.argument_dict[event_type.replace(':', '.')]
            # NOTE: Large change - original only keep one if multiple span for one arg role
            arg_quries = dict()
            arg_joint_prompt = dict()
            target_info = dict()
            if os.environ.get("DEBUG", False): arg_set=set()
            for arg in arg_list:
                arg_query = None
                prompt_slots = None
                arg_target = {
                    "text": list(),
                    "span_s": list(),
                    "span_e": list()
                }

                if self.arg_query:
                    arg_query = self.create_dec_qury(arg, event_trigger[0])
                if self.prompt_query:
                    prompt_slots = {
                        "tok_s":list(), "tok_e":list(),
                    }
                    
                    # Using this more accurate regular expression might further improve rams results
                    for matching_result in re.finditer(r'\b'+re.escape(arg)+r'\b', dec_prompt_text.split('.')[0]):
                        char_idx_s, char_idx_e = matching_result.span(); char_idx_e -= 1
                        tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
                        tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
                        prompt_slots["tok_s"].append(tok_prompt_s);prompt_slots["tok_e"].append(tok_prompt_e)
                answer_texts, start_positions, end_positions = list(), list(), list()
                if arg in event_args_name:
                    # Deal with multi-occurance
                    if os.environ.get("DEBUG", False): arg_set.add(arg)
                    arg_idxs = [i for i, x in enumerate(event_args_name) if x == arg]
                    if os.environ.get("DEBUG", False): counter[0] += 1; counter[1]+=len(arg_idxs)

                    for arg_idx in arg_idxs:
                        event_arg_info = event_args[arg_idx]
                        answer_text = event_arg_info['text']; answer_texts.append(answer_text)
                        start_old, end_old = event_arg_info['start'], event_arg_info['end']
                        start_position = old_tok_to_new_tok_index[start_old][0]; start_positions.append(start_position)
                        end_position = old_tok_to_new_tok_index[end_old-1][1]; end_positions.append(end_position)

                if self.arg_query:
                    arg_target["span_s"] = [1 if i in start_positions else 0 for i in range(self.args.max_enc_seq_length)]
                    arg_target["span_e"] = [1 if i in end_positions else 0 for i in range(self.args.max_enc_seq_length)]
                    if sum(arg_target["span_s"])==0:
                        arg_target["span_s"][0] = 1
                        arg_target["span_e"][0] = 1
                if self.prompt_query:
                    arg_target["span_s"]= start_positions
                    arg_target["span_e"] = end_positions

                arg_target["text"] = answer_texts
                arg_quries[arg] = arg_query
                arg_joint_prompt[arg] = prompt_slots
                target_info[arg] = arg_target

            if not self.arg_query:
                arg_quries = None
            if not self.prompt_query:
                arg_joint_prompt=None

            # NOTE: one annotation as one decoding input
            feature_idx = len(features)
            features.append(
                    InputFeatures(example_id, feature_idx,
                                event_type, event_trigger,
                                enc_text, enc_input_ids, enc_mask_ids,
                                dec_prompt_text, dec_prompt_ids, dec_prompt_mask_ids,
                                arg_quries, arg_joint_prompt, target_info,
                                old_tok_to_new_tok_index = old_tok_to_new_tok_index, full_text=example.full_text, arg_list = arg_list,
                    )
            )

        if os.environ.get("DEBUG", False): print('\033[91m'+f"distinct/tot arg_role: {counter[0]}/{counter[1]} ({counter[2]})"+'\033[0m')
        return features
    def convert_features_to_dataset(self, features):
        dataset = ArgumentExtractionDataset(features)
        return dataset