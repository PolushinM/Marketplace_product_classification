import re
import json

import torch
import torch.nn as nn


def cleanhtml(raw_html):
    cleantext = re.sub('<.*?>', ' ', raw_html)
    return cleantext

def repr_field(s: object) -> str:
    s = cleanhtml(repr(s))
    s = s.replace('[', ' ').replace(']', ' ')
    s = s.replace('{', ' ').replace('}', ' ')
    s = s.replace("'", '')
    return s

def get_title(field: str) -> str:
    field = json.loads(field)
    title = cleanhtml(str(field['title']))
    return title

def preprocess_text_field(field: str) -> str:
    field = json.loads(field)
    title = cleanhtml(str(field['title']))
    description = cleanhtml(str(field['description']))
    attributes = repr_field(field['attributes'])
    custom_characteristics = repr_field(field['custom_characteristics'])
    defined_characteristics = repr_field(field['defined_characteristics'])
    filters = repr_field(field['filters'])
    return ". ".join([description, attributes, custom_characteristics, defined_characteristics, filters])


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class Attention(nn.Module):
    def __init__(self, query_dim, value_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(query_dim, value_dim, bias=False)
    def forward(self, query_emb, value_emb):
        attention = torch.sigmoid(self.fc(query_emb))
        return value_emb * attention