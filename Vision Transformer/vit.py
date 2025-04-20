import torch 
import torch.nn as nn
import math

class PatchEmbeddings(nn.Module):

    def __init__(self,image_size,patch_size,in_channels,hidden_size):
        super(PatchEmbeddings,self).__init__()

        num_patches = (image_size // patch_size)**2
        self.conv = nn.Conv2d(in_channels,hidden_size,kernel_size = patch_size,stride=patch_size)

    def forward(self,x):
        x =  self.conv(x)
        x = x.flatten(2)
        x  = x.transpose(1,2)
        return x

class Embeddings(nn.Module):

    def __init__(self,image_size,patch_size,in_channels,hidden_size,dropout_prob):
        super(Embeddings,self).__init__()

        num_patches = (image_size // patch_size)**2
        
        self.patch_embeddings = PatchEmbeddings(image_size,patch_size,in_channels,hidden_size)
        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1,1,hidden_size))
        # Learnable Position Embeddings
        self.position_embedding = nn.Parameter(torch.randn(1,num_patches+1,hidden_size)) 
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,x):
        # Input size (batch_size,C,H,W)
        x = self.patch_embeddings(x)
        # Size (batch_size,H*W,hidden_size)
        batch_size,_,_ = x.size()
        # Extend cls_tokens across all batches
        cls_tokens = self.cls_token.expand(batch_size,-1,-1)
        # Size (batch_size,1,hidden_size)
        x = torch.cat((x,cls_tokens),dim=1)
        # Size (batch_size,1+H*W,hidden_size)
        x = x + self.position_embedding
        # Size (batch_size,1+H*W,hidden_size)
        return self.dropout(x)


class SelfAttention(nn.Module):

    def __init__(self, hidden_size, attention_head_size, dropout_prob, bias=True):
        super(SelfAttention,self).__init__()
        self.hidden_size = hidden_size
        self.dk = attention_head_size

        self.query_projection = nn.Linear(hidden_size,attention_head_size,bias)
        self.key_projection = nn.Linear(hidden_size,attention_head_size,bias)
        self.value_projection = nn.Linear(hidden_size,attention_head_size,bias)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self,x):
        # Size : (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length, dk)
        key = self.key_projection(x)
        query = self.query_projection(x)
        value = self.value_projection(x)

        # softmax(Q*K.T/sqrt(dk))*V
        attention_scores = (query @ key.transpose(1,2))/math.sqrt(self.dk)
        # size (batch_size,sequence_length,sequence_length)
        
        attention_scores = nn.functional.softmax(attention_scores,dim = -1)
        attention_scores = self.dropout(attention_scores)

        output = attention_scores @ value
        # size (batch_size,sequence_length,dk)
        return output,attention_scores

class MultliHeadSelfAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, dropout_prob,bias=True) :
        super(MultliHeadSelfAttention).__init__()

        self.hidden_size = hidden_size
        self.dk = self.hidden_size // num_heads

        self.heads = nn.ModuleList([])
        for _ in range(num_heads):
            head = SelfAttention(self.hidden_size, self.dk, dropout_prob, bias=True)
            self.heads.append(head)
        self.dropout = nn.Dropout(dropout_prob)
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(dropout_prob)

        def forward(self,x):
            attn_outputs = head(x) 
            return
        
class MHA(nn.Module):

    def __init__(self,hidden_size,num_heads, dropout_prob,bias = True):
        super(MHA,self).__init__()

        self.hidden_size = hidden_size
        self.dk = self.hidden_size // num_heads
        self.num_heads = num_heads

        self.query_projection = nn.Linear(hidden_size,hidden_size,bias)
        self.key_projection = nn.Linear(hidden_size,hidden_size,bias)
        self.value_projection = nn.Linear(hidden_size,hidden_size,bias)
        
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self,x):
        ## x : (batch_size,seq_len,hidden_size)
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)
        ## q,k,v : (batch_size,seq_len,hidden_size)
        batch_size,seq_len ,_ = query.size()
        ### q,k,v convert to (batch_size,seq_len,num_heads,dk) => (batch_size,num_heads,seq_len,dk)
        # dk = hidden_size/num_heads
        query = query.view(batch_size,seq_len,self.num_heads,self.dk).transpose(1,2)
        value = value.view(batch_size,seq_len,self.num_heads,self.dk).transpose(1,2)
        key = key.view(batch_size,seq_len,self.num_heads,self.dk).transpose(1,2)

        # softmax(Q*K.T/sqrt(dk))*V
        attention_scores = (query @ key.transpose(-1,-2))/math.sqrt(self.dk)
        # size (batch_size,num_heads,seq_len,seq_len)
        attention_scores = nn.functional.softmax(attention_scores,dim = -1)
        attention_scores = self.dropout(attention_scores)

        output = attention_scores @ value
        ## size (batch_size,num_heads,seq_len,dk) => (batch_size,seq_len,num_heads,dk) => (batch_size,seq_len,hidden_size)
        output = output.transpose(1,2).contiguous().view(batch_size,seq_len,self.hidden_size)
        output = self.dropout(output)

        return output, attention_scores

class MLP(nn.Module):

    def __init__(self, hidden_size = 48, intermediate_size = 4*48,dropout_prob = 0.2) :
        super(MLP,self).__init__()

        self.fc1 = nn.Linear(hidden_size,intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()
    def forward(self,x):
        return self.dropout(self.fc2(self.activation(self.fc1(x))))
    
class TransformerEncoderBlock(nn.Module):

    def __init__(self, hidden_size,num_heads,dropout_prob,bias = True) -> None:
        super(TransformerEncoderBlock,self).__init__()

        self.layerNorm = nn.LayerNorm(hidden_size)
        self.MutliHeadAttention = MHA(hidden_size,num_heads,dropout_prob,bias)
        self.mlp = MLP(hidden_size, 4*hidden_size,dropout_prob)

    def forward(self,x):

        ## Layer Normalization and MHA Block
        att_output, att_scores = self.MutliHeadAttention(self.layerNorm(x))

        ## Skip Connection 
        x = (x + att_output)
        ## Layer Normalization and MLP Block
        mlp_output = self.mlp(self.layerNorm(x))

        ##Skip Connection
        x = (x + mlp_output)

        return x,att_scores
    

class ViT(nn.Module):

    def __init__(self,image_size,patch_size,in_channels,num_classes,num_encoder_layers,hidden_size,num_heads,dropout_prob,bias = True):
        super(ViT,self).__init__()

        self.embedding_layer = Embeddings(image_size,patch_size,in_channels,hidden_size,dropout_prob)
        self.encoder_blocks = nn.ModuleList([])
    

        for _ in range(num_encoder_layers):
            block = TransformerEncoderBlock(hidden_size,num_heads,dropout_prob,bias)
            self.encoder_blocks.append(block)

        self.classifier = nn.Linear(hidden_size,num_classes)
        # self.apply(self._init_weights)

    def forward(self,x):
        att_scores_list = []
        x = self.embedding_layer(x)
        print(x.shape)
        for block in self.encoder_blocks:
            x, att_scores = block(x)
            print(att_scores.shape)
            att_scores_list.append(att_scores)
        print(x.shape)
        output = self.classifier(x[:, 0, :])
        print(output.shape)
        return output,att_scores_list




def main():
    # model1 = PatchEmbeddings(image_size=224,patch_size=16,in_channels=3,hidden_size=48)
    # model2 = Embeddings(image_size=224,patch_size=16,in_channels=3,hidden_size=48,dropout_prob=0.2)
    # # model3 = SelfAttention(hidden_size = 48, attention_head_size = 12, dropout_prob =0.2, bias=True)
    # # model3 = MSA(hidden_size = 48,num_heads = 4, dropout_prob =0.2,bias = True)

    # model4 = TransformerBlock(hidden_size=48,num_heads=4,dropout_prob=0.2, bias=True)
    x = torch.randn(1,3,224,224)
    # output = model1(x)
    # output= model2(x)
    # print(output.shape)
    # output,_ = model4(output)

    # print(output.shape)
    model = ViT(image_size = 224,patch_size = 16,in_channels = 3,num_classes = 10,num_encoder_layers = 6,hidden_size = 48,num_heads = 4,dropout_prob = 0.1,bias = True)
    output,_ = model(x)
    print(model)
    print(output.shape)
if __name__ == "__main__":
    main()



    

