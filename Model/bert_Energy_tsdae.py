from transformers import BertTokenizer
from transformers import AutoModel


import torch


class BertEnergy(object):
    def __init__(self, model_name='ontology/EnergyBert'):
        '''
        sbert_energy_stsb
        '''

        self.tokenizer = BertTokenizer.from_pretrained('ontology/EnergyBert')
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def embedding(self, sentences):
        tokens = {'input_ids': [], 'attention_mask': []}
        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=20,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
            tokens['input_ids'].append(encoded_dict['input_ids'])
            tokens['attention_mask'].append(encoded_dict['attention_mask'])
        tokens['input_ids'] = torch.cat(tokens['input_ids'], dim=0)
        tokens['attention_mask'] = torch.cat(tokens['attention_mask'], dim=0)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**tokens)
        embeddings = outputs.last_hidden_state
        attention = tokens['attention_mask']
        mask = attention.unsqueeze(-1).expand(embeddings.shape).float()
        mask_embeddings = embeddings * mask
        summed = torch.sum(mask_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts
        return mean_pooled


