import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('Device:', 'cuda:0')
# print('Current cuda device:', torch.cuda.current_device())
# print('Count of using GPUs:', torch.cuda.device_count())
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

sentences1 = ["This is an example sentence","Each sentence is converted"]
sentences2 = ["Each sentence is converted","Each sentence is converted"]

# input dim [256]
encoded_input = tokenizer(sentences1, padding=True, truncation=True, max_length=256, return_tensors='pt')
encoded_target = tokenizer(sentences2, padding=True, truncation=True, max_length=256, return_tensors='pt')

class PyTorch_to_TorchScript(torch.nn.Module):
    def __init__(self):
        super(PyTorch_to_TorchScript, self).__init__()
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').cuda()
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0].cuda() #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float().cuda()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9).cuda()

    def forward(self, input_ids, token_type_ids, attention_mask):
        model_output = self.model(input_ids=input_ids.cuda(), attention_mask=attention_mask.cuda(), token_type_ids=token_type_ids.cuda())
        sentence_embeddings = self.mean_pooling(model_output, attention_mask).cuda()
        return F.normalize(sentence_embeddings, p=2, dim=1).cuda()
    

pt_model = PyTorch_to_TorchScript().eval()
input_ids = encoded_input['input_ids']
print(input_ids.dtype)
token_type_ids = encoded_input['token_type_ids']
print(token_type_ids.shape)
print(token_type_ids.dtype)
attention_mask = encoded_input['attention_mask']
print(attention_mask.shape)
print(attention_mask.dtype)
res = pt_model(input_ids, token_type_ids, attention_mask)
print(res.shape)
print(res.dtype)
# output dim [384]
traced_script_module = torch.jit.trace(pt_model, (input_ids, token_type_ids, attention_mask), strict=False)
traced_script_module.save("model_sbert.pt")