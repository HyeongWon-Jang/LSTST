import torch
import torch.nn as nn

from transformers import GPTJConfig, GPTJForCausalLM,AutoTokenizer
from layers.Gpt_embed import DataEmbedding
feat_dim_dict = {
    'EthanolConcentration': 3,
    'FaceDetection': 144,
    'Handwriting': 3,
    'Heartbeat': 61,
    'JapaneseVowels': 12,
    'PEMS-SF': 963,
    'SelfRegulationSCP1': 6,
    'SelfRegulationSCP2': 7,
    'SpokenArabicDigits': 13,
    'UWaveGestureLibrary': 3,
}

num_classes_dict = {
    'EthanolConcentration': 4,
    'FaceDetection': 2,
    'Handwriting': 26,
    'Heartbeat': 2,
    'JapaneseVowels': 9,
    'PEMS-SF': 7,
    'SelfRegulationSCP1': 2,
    'SelfRegulationSCP2': 2,
    'SpokenArabicDigits': 10,
    'UWaveGestureLibrary': 8,
}

max_len_dict = {
    'EthanolConcentration': 1751,
    'FaceDetection': 62,
    'Handwriting': 152,
    'Heartbeat': 405,
    'JapaneseVowels': 29,
    'PEMS-SF': 144,
    'SelfRegulationSCP1': 896,
    'SelfRegulationSCP2': 1152,
    'SpokenArabicDigits': 93,
    'UWaveGestureLibrary': 315,
}

def get_class_index(num_classes, tokenizer):
    class_index = []
    for i in range(num_classes):
        class_index.extend(tokenizer.encode(str(i)))
    return class_index
    
    

def get_text_ids(tokenizer, max_len, kernel_width, stride, padding, num_classes):
    class_index = [str(i) for i in range(num_classes)]
    class_text = ", ".join(class_index)
    text = "Time series data exists with "
    
    patch_num =  (max_len + 2* padding - kernel_width) // stride + 1
    time = list(range(1, patch_num+1))
    
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
    def __init__(self, configs, llm="EleutherAI/gpt-j-6B"):
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

        self.gptj_config = GPTJConfig.from_pretrained('EleutherAI/gpt-j-6B')

        self.gptj_config.output_attentions = True
        self.gptj_config.output_hidden_states = True
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, 
                                             bnb_4bit_use_double_quant=True,
                                             bnb_4bit_quant_type="nf4",
                                             bnb_4bit_compute_dtype=torch.bfloat16,
                                             llm_int8_skip_modules=["lm_head"])
        
        self.llm_model = GPTJForCausalLM.from_pretrained(
                llm,
                quantization_config = quantization_config,
                config=self.gptj_config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
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
            if 'ln' in name or 'wpe' in name or 'head' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

       

        self.word_embeddings = self.llm_model.get_input_embeddings()
        self.text_ids, self.times = get_text_ids(self.tokenizer, self.max_len, self.kernel_width, self.stride, self.padding, self.num_classes)
        self.ts_embedding = DataEmbedding(
            self.feat_dim, self.d_llm, self.d_llm, len(self.times), dropout=configs.dropout, kernel_width=self.kernel_width, stride=self.stride,padding=self.padding
        )
        
        self.special_token_id = self.tokenizer.additional_special_tokens_ids[0]
        self.classifier = nn.Linear(self.d_llm, self.num_classes)
        class_index = get_class_index(self.num_classes, self.tokenizer)
        
        with torch.no_grad():
            self.classifier.weight.copy_(self.llm_model.lm_head.weight[class_index, :])
            self.classifier.bias.copy_(self.llm_model.lm_head.bias[class_index])
        self.llm_model.lm_head.weight.requires_grad = False
        self.llm_model.lm_head.bias.requires_grad = False
        for name, param in self.named_parameters():
            if param.requires_grad ==True:
                print(name)
                pass


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
        
        outputs = self.llm_model(inputs_embeds=inputs_embeds, return_dict=True).hidden_states[-1]
        pred = self.classifier(outputs[:, -1, :])
        return pred