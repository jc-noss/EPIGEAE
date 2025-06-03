# paie model
import json

import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel
from utils import hungarian_matcher, get_best_span, get_best_span_simple
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F
from .graph_attention_layers import GATLayer

BERT_PATH = './ckpts/bert-base-uncased'
tokenizer_bert = BertTokenizer.from_pretrained(BERT_PATH)
bert_model = BertModel.from_pretrained(BERT_PATH)



def l2_normalize(tensor):
    norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
    normalized_tensor = tensor / (norm + 1e-8)
    return normalized_tensor

class EPIG_EAE(BartPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = BartModel(config)
        self.w_prompt_start = nn.Parameter(torch.rand(config.d_model, ))
        self.w_prompt_end = nn.Parameter(torch.rand(config.d_model, ))
        self.model._init_weights(self.w_prompt_start)
        self.model._init_weights(self.w_prompt_end)
        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')
        self.gat_layer = GATLayer(config, config.d_model, config.d_model,config.d_model)
        self.edge_update_matrix = nn.Parameter(torch.rand(config.d_model, config.d_model))
        self.shared_role_matrix = nn.Parameter(torch.rand(config.d_model, config.d_model))
        self.model._init_weights(self.edge_update_matrix)
        self.model._init_weights(self.shared_role_matrix)
        if self.config.dataset_type == 'rams':
            role_path='./data/dset_meta/role_num_rams.json'
        elif self.config.dataset_type == 'wikievent':
            role_path='./data/dset_meta/role_num_wikievent.json'
        elif self.config.dataset_type == 'oeecfc':
            role_path='./data/dset_meta/dict_role_OEE_CFC.json'
        self.role_embeddings = {}
        with open(role_path, 'r') as f:
            self.role_dict = json.load(f)
        for event_type, roles in self.role_dict.items():
            self.role_embeddings[event_type] = {}
            for role in roles:
                self.role_embeddings[event_type][role] = None

    def get_bert_embedding(self,text):
        if text:
            inputs = tokenizer_bert(text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().to(self.config.device)
        else:
            return torch.empty(self.config.hidden_size)
    def extract_argsFeatures_from_contextOutput(self, predicted_spans, context_output):
        extracted_args = []
        for span in predicted_spans:
            if(self.config.matching_method_train=='max'):
                start, end = span[0].item(), span[1].item()
            elif self.config.matching_method_train == 'accurate':
                start, end = span[0], span[1]
            arg_features = torch.mean(context_output[start:end+1], dim=0)
            extracted_args.append(arg_features)
        return extracted_args
    def forward(
        self,
        enc_input_ids=None,
        enc_mask_ids=None,
        dec_prompt_ids=None,
        dec_prompt_mask_ids=None,
        arg_joint_prompts=None,
        target_info=None,
        old_tok_to_new_tok_indexs=None,
        arg_list=None,
        event_trigger=None,
        event_type=None,
        full_text=None,
    ):
        if self.config.context_representation == 'decoder':
            context_outputs = self.model(
                enc_input_ids,
                attention_mask=enc_mask_ids,
                return_dict=True,
            )
            decoder_context = context_outputs.encoder_last_hidden_state
            context_outputs = context_outputs.last_hidden_state
        else:
            context_outputs = self.model.encoder(
                enc_input_ids,
                attention_mask=enc_mask_ids,
            )
            context_outputs = context_outputs.last_hidden_state
            decoder_context = context_outputs
        decoder_prompt_outputs = self.model.decoder(
                input_ids=dec_prompt_ids,
                attention_mask=dec_prompt_mask_ids,
                encoder_hidden_states=decoder_context,
                encoder_attention_mask=enc_mask_ids,
        )
        decoder_prompt_outputs = decoder_prompt_outputs.last_hidden_state   #[bs, prompt_len, H]
        total_loss = 0
        logit_lists = []
        for i, (context_output, decoder_prompt_output, arg_joint_prompt, old_tok_to_new_tok_index) in \
                enumerate(zip(context_outputs, decoder_prompt_outputs, arg_joint_prompts, old_tok_to_new_tok_indexs)):
            output = dict()
            predicted_spans = []
            for arg_role in arg_joint_prompt.keys():
                prompt_slots = arg_joint_prompt[arg_role]
                start_logits_list = []
                end_logits_list = []
                for (p_start, p_end) in zip(prompt_slots['tok_s'], prompt_slots['tok_e']):
                    prompt_query_sub = decoder_prompt_output[p_start:p_end]
                    prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)
                    start_query = (prompt_query_sub * self.w_prompt_start).unsqueeze(-1)  # [1, H, 1]
                    end_query = (prompt_query_sub * self.w_prompt_end).unsqueeze(-1)  # [1, H, 1]
                    start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()
                    end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()
                    start_logits_list.append(start_logits)
                    end_logits_list.append(end_logits)
                output[arg_role] = [start_logits_list, end_logits_list]
                for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                    if self.config.matching_method_train == 'accurate':
                        predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index,
                                                             self.config.max_span_length))
                    elif self.config.matching_method_train == 'max':
                        predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                    else:
                        raise AssertionError()
            for iterEpoch in range(self.config.iterEpoch_num):
                trigger = event_trigger[i][0]
                etype = event_type[i]
                args=self.extract_argsFeatures_from_contextOutput(predicted_spans,context_output)
                attrs = []
                for arg_role in arg_joint_prompt.keys():
                    prompt_slots = arg_joint_prompt[arg_role]
                    attrs.extend([arg_role] * len(prompt_slots['tok_s']))
                trigger_emb = self.get_bert_embedding(trigger)
                etype_emb = self.get_bert_embedding(etype)
                args_emb = [arg for arg in args]
                attrs_emb = [self.get_bert_embedding(attr) for attr in attrs]
                x = torch.stack([trigger_emb] + args_emb + [etype_emb] + attrs_emb)
                edge_index = []
                edge_attr = []
                for k, role in enumerate(args):#tri-args
                    edge_index.append([0, k + 1])
                    edge_index.append([k + 1, 0])
                    edge_attr.append(attrs_emb[k])
                    edge_attr.append(attrs_emb[k])
                for k, attr in enumerate(attrs):#etype-roles
                    edge_index.append([len(args) + 1, len(args) + 2 + k])
                    edge_index.append([len(args) + 2 + k, len(args) + 1])
                    edge_attr.append(self.get_bert_embedding('属性'))
                    edge_attr.append(self.get_bert_embedding('属性'))
                edge_index.append([0, len(args) + 1])#tri-etype
                edge_index.append([len(args) + 1, 0])
                edge_attr.append(self.get_bert_embedding('实例'))
                edge_attr.append(self.get_bert_embedding('实例'))
                for k, arg in enumerate(args):#args-roles
                    edge_index.append([k + 1, len(args) + 2 + k])
                    edge_index.append([len(args) + 2 + k, k + 1])
                    edge_attr.append(self.get_bert_embedding('值'))
                    edge_attr.append(self.get_bert_embedding('值'))
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.stack(edge_attr)
                updated_x=x
                for _ in range(2):
                    updated_x = self.gat_layer(updated_x, edge_index,edge_attr)
                    for j, arg in enumerate(args):
                        trigger_to_arg_idx = 2 * j
                        arg_to_trigger_idx = 2 * j + 1
                        arg_node_index = j + 1
                        attr_node_index = 2+len(args)+j
                        new_edge_emb_trigger_to_arg = (edge_attr[trigger_to_arg_idx, :] + updated_x[0] + updated_x[arg_node_index] +
                                                       updated_x[attr_node_index]) / 4
                        new_edge_emb_arg_to_trigger = (edge_attr[arg_to_trigger_idx, :] + updated_x[0] + updated_x[arg_node_index] +
                                                       updated_x[attr_node_index]) / 4
                        new_edge_emb_trigger_to_arg=new_edge_emb_trigger_to_arg.to(self.config.device)
                        new_edge_emb_arg_to_trigger=new_edge_emb_arg_to_trigger.to(self.config.device)
                        new_edge_emb_trigger_to_arg_adjusted = torch.matmul(new_edge_emb_trigger_to_arg.unsqueeze(0),self.edge_update_matrix).squeeze()
                        new_edge_emb_arg_to_trigger_adjusted = torch.matmul(new_edge_emb_arg_to_trigger.unsqueeze(0),self.edge_update_matrix).squeeze()
                        new_edge_emb_trigger_to_arg_adjusted = F.gelu(new_edge_emb_trigger_to_arg_adjusted)
                        new_edge_emb_arg_to_trigger_adjusted = F.gelu(new_edge_emb_arg_to_trigger_adjusted)
                        edge_attr[trigger_to_arg_idx, :] = new_edge_emb_trigger_to_arg_adjusted
                        edge_attr[arg_to_trigger_idx, :] = new_edge_emb_arg_to_trigger_adjusted
                updated_x_new = updated_x.clone()
                for k, attr in enumerate(attrs):
                    attr_node_index = len(args) + 2 + k
                    attr_edge_index = 2 * k
                    edge_emb = edge_attr[attr_edge_index, :]
                    avg_attr_node_edge_emb = (updated_x_new[attr_node_index, :] + edge_emb) / 2
                    updated_x_new[attr_node_index, :] = avg_attr_node_edge_emb
                updated_x = updated_x_new
                batch_loss = list()
                cnt = 0
                output = dict()
                num_args = len(args)
                start_attr_index = 2 + num_args
                j=0
                predicted_spans = []
                for _, arg_role in enumerate(arg_joint_prompt.keys()):
                    prompt_slots = arg_joint_prompt[arg_role]
                    start_logits_list = []
                    end_logits_list = []
                    for (p_start, p_end) in zip(prompt_slots['tok_s'], prompt_slots['tok_e']):
                        prompt_query_sub_origin = decoder_prompt_output[p_start:p_end]
                        prompt_query_sub_origin = torch.mean(prompt_query_sub_origin, dim=0).unsqueeze(0)
                        attr_index = start_attr_index + j
                        prompt_query_sub_attr = updated_x[attr_index].unsqueeze(0)
                        prompt_query_sub_attr = prompt_query_sub_attr.to(self.config.device)
                        j=j+1
                        prompt_query_sub_origin_normalized = l2_normalize(prompt_query_sub_origin)
                        prompt_query_sub_attr_normalized = l2_normalize(prompt_query_sub_attr)
                        if self.role_embeddings[etype][arg_role] is None:
                            if self.training:
                                self.role_embeddings[etype][arg_role] = (prompt_query_sub_origin_normalized + prompt_query_sub_attr_normalized) / 2
                            prompt_query_sub = (prompt_query_sub_origin_normalized + prompt_query_sub_attr_normalized) / 2
                        else:
                            role_embedding_normalized = l2_normalize(self.role_embeddings[etype][arg_role].detach())
                            prompt_query_sub = (prompt_query_sub_origin_normalized + prompt_query_sub_attr_normalized + role_embedding_normalized) / 3
                            new_value = torch.matmul(prompt_query_sub.detach(), self.shared_role_matrix)
                            if self.training:
                                self.role_embeddings[etype][arg_role] = new_value
                        start_query = (prompt_query_sub * self.w_prompt_start).unsqueeze(-1)  # [1, H, 1]
                        end_query = (prompt_query_sub * self.w_prompt_end).unsqueeze(-1)  # [1, H, 1]
                        start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()
                        end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()
                        start_logits_list.append(start_logits)
                        end_logits_list.append(end_logits)
                    output[arg_role] = [start_logits_list, end_logits_list]
                    if (iterEpoch == self.config.iterEpoch_num - 1):
                        if self.training:
                            target = target_info[i][arg_role]
                            predicted_spans = []
                            for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                                if self.config.matching_method_train == 'accurate':
                                    predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index,self.config.max_span_length))
                                elif self.config.matching_method_train == 'max':
                                    predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                                else:
                                    raise AssertionError()
                            target_spans = [[s, e] for (s, e) in zip(target["span_s"], target["span_e"])]
                            if len(target_spans) < len(predicted_spans):
                                pad_len = len(predicted_spans) - len(target_spans)
                                target_spans = target_spans + [[0, 0]] * pad_len
                                target["span_s"] = target["span_s"] + [0] * pad_len
                                target["span_e"] = target["span_e"] + [0] * pad_len
                            if self.config.bipartite:
                                idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)
                            else:
                                idx_preds = list(range(len(predicted_spans)))
                                idx_targets = list(range(len(target_spans)))
                                if len(idx_targets) > len(idx_preds):
                                    idx_targets = idx_targets[0:len(idx_preds)]
                                idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                                idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)
                            cnt += len(idx_preds)
                            start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds],torch.LongTensor(target["span_s"]).to(self.config.device)[idx_targets])
                            end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds],torch.LongTensor(target["span_e"]).to(self.config.device)[idx_targets])
                            batch_loss.append((start_loss + end_loss) / 2)
                    else:
                        for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                            if self.config.matching_method_train == 'accurate':
                                predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index,
                                                                     self.config.max_span_length))
                            elif self.config.matching_method_train == 'max':
                                predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                            else:
                                raise AssertionError()
            logit_lists.append(output)
            if self.training:  # inside batch mean loss
                total_loss = total_loss + torch.sum(torch.stack(batch_loss)) / cnt
        if self.training:
            return total_loss / len(context_outputs), logit_lists
        else:
            return [], logit_lists
