import torch
import torch.nn as nn

from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from layers.Gpt_embed import DataEmbedding
feat_dim_dict = {
    'EthanolConcentration': 3,
    'FaceDetection ': 144,
    'Handwriting ': 3,
    'Heartbeat': 61,
    'JapaneseVowels': 12,
    'PEMS-SF ': 963,
    'SelfRegulationSCP1': 6,
    'SelfRegulationSCP2': 7,
    'SpokenArabicDigits': 13,
    'UWaveGestureLibrary': 3,
}

num_classes_dict = {
    'EthanolConcentration': 4,
    'FaceDetection ': 2,
    'Handwriting ': 26,
    'Heartbeat': 2,
    'JapaneseVowels': 9,
    'PEMS-SF ': 7,
    'SelfRegulationSCP1': 2,
    'SelfRegulationSCP2': 2,
    'SpokenArabicDigits': 10,
    'UWaveGestureLibrary': 8,
}

max_len_dict = {
    'EthanolConcentration': 1751,
    'FaceDetection ': 62,
    'Handwriting ': 152,
    'Heartbeat': 405,
    'JapaneseVowels': 29,
    'PEMS-SF ': 144,
    'SelfRegulationSCP1': 896,
    'SelfRegulationSCP2': 1152,
    'SpokenArabicDigits': 93,
    'UWaveGestureLibrary': 315,
}

def get_text_ids(tokenizer, max_len, kernel_width, stride, padding, num_classes):
    class_index = [str(i) for i in range(num_classes)]
    class_text = ", ".join(class_index)
    text = "Time series data exists with "
    time = list(range(1, max_len+1))
    
    start_point = kernel_width // 2 - padding
    time = time[start_point::stride]
    lists = ", ".join(
            sum(
                [
                    [f"patch{i+1} is <|patch|>"]
                    for i,t in enumerate(time)
                ],
                [],
            )
    )
    text = text + lists + ". What is label? Please choose in [" + class_text + "] A: "
    
    text_ids = tokenizer.encode(text)
    
    return text_ids, time
    

class Model(nn.Module):
    def __init__(self, configs, llm="gpt2"):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.data = configs.model_id
        
        self.d_llm = configs.llm_dim
        self.kernel_width = configs.kernel_width
        self.stride = configs.stride
        self.padding = configs.padding
        
        self.max_len = max_len_dict[self.data]
        self.num_classes = num_classes_dict[self.data]
        self.feat_dim = feat_dim_dict[self.data]

        self.gpt2_config = GPT2Config.from_pretrained('gpt2')

        self.gpt2_config.output_attentions = True
        self.gpt2_config.output_hidden_states = True
        self.llm_model = GPT2Model.from_pretrained(
                llm,
                cache_dir="/mnt/aitrics_ext/ext01/kaist_guest/kaist/timeseries/huggingface_cache",
                offload_state_dict=True,
                offload_folder="./offload_folder/",
                trust_remote_code=True,
                config=self.gpt2_config,
        )

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            llm,
            trust_remote_code=True
        )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<|patch|>']})
        
        for name, param in self.llm_model.named_parameters():
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.ts_embedding = DataEmbedding(
            self.feat_dim, self.d_llm, self.d_llm, self.max_len, dropout=configs.dropout, kernel_width=self.kernel_width, stride=self.stride,padding=self.padding
        )

        self.word_embeddings = self.llm_model.get_input_embeddings()
        self.text_ids, self.times = get_text_ids(self.tokenizer, self.max_len, self.kernel_width, self.stride, self.padding, self.num_classes)
        self.special_token_id = self.tokenizer.additional_special_tokens_ids[0]
        self.classifier = nn.Linear(self.d_llm, self.num_classes)
        for name, param in self.named_parameters():
            if param.requires_grad ==True:
                print(name)


    def forward(self, x_enc, x_mark_enc, x_dec, x_dec_mark):
        B, L, M = x_enc.shape
        # text_embedding
        text_ids = torch.LongTensor([self.text_ids] * B).to(x_enc.device)
        
        special_mask = (text_ids != self.special_token_id)
        special_mask = special_mask.unsqueeze(dim=2).repeat(1,1,self.d_llm)

        text_ids[text_ids  == self.special_token_id] = 0
        inputs_embeds = self.word_embeddings(text_ids) * special_mask
        
        # time series embedding
        times = torch.LongTensor(self.times).to(x_enc.device)
        ts_embed = self.ts_embedding(x_enc, times)
        inputs_embeds[special_mask == False] = ts_embed.reshape(-1)
        
        outputs = self.llm_model(inputs_embeds=inputs_embeds).last_hidden_state
        pred = self.classifier(outputs[:, -1, :])
        return pred