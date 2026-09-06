from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from models.dsee import create_vit, init_tokenizer, load_checkpoint

from collections import OrderedDict

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    """https://github.com/woodfrog/vse_infty, thanks!"""
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) from https://github.com/woodfrog/vse_infty, thanks!"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                    #  768, 512, 1024, 2
        super().__init__()
        self.output_dim = output_dim

        self.layer1 = nn.Linear(input_dim, hidden_dim)          # 768, 512
        self.bn1 = nn.BatchNorm1d(hidden_dim)                   # 512

        self.layer2 = nn.Linear(hidden_dim, output_dim)         # 512, 1024
        # self.bn2 = nn.BatchNorm1d(output_dim)                   # 1024

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)

        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = x.view(B, N, self.output_dim)
        return x


class TexualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=768, embed_dim=256, ratio=0.3):
        super(TexualEmbeddingLayer, self).__init__()
        self.embed_dim= embed_dim
        self.ratio = ratio
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)


    def forward(self, features, token, gated_atten):
        mask =  ((token != 0) + 0)
        lengths = mask.sum(1).view(-1) - 2 
        k = int((token.size(1)-2)*self.ratio)  
        bs = features.size(0)

        atten = gated_atten.clone() * mask
        atten[torch.arange(bs), 0] = -1                
        atten[torch.arange(bs), mask.sum(1) - 1] = -1           

        atten_topK = atten.topk(dim=-1,k = k)[1].unsqueeze(-1).expand(bs,k,features.size(2)) # bs x k x hidden
        features = torch.gather(input=features,dim=1,index=atten_topK)  # bs x k x hidden
        features = l2norm(features, dim=-1)

        lengths = torch.clamp(lengths, max=k)  

        cap_emb = self.fc(features)
        features = self.mlp(features) + cap_emb   
        features = maxk_pool1d_var(features, 1, 1, lengths.to(cap_emb.device)) 

        return features


class VisualEmbeddingLayer(nn.Module):
    def __init__(self, input_dim=768, embed_dim=256, ratio=0.3):
        super(VisualEmbeddingLayer, self).__init__()
        self.embed_dim= embed_dim
        self.ratio = ratio
        self.fc = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)

    def forward(self, base_features, gated_atten):
        k = int((gated_atten.size(1)-1)*self.ratio)  

        bs = base_features.size(0)
        atten = gated_atten.clone()
        atten[:, 0] = -1                       

        atten_topK = atten.topk(dim=-1,k = k)[1]
        atten_topK = atten_topK.unsqueeze(-1).expand(bs, k, base_features.size(2))
        base_features = torch.gather(input=base_features,dim=1,index=atten_topK)
        base_features = l2norm(base_features, dim=-1)
        feat_lengths = torch.full((bs,), k, device=base_features.device)

        img_features = self.fc(base_features)
        features = self.mlp(base_features) + img_features

        features = maxk_pool1d_var(features, 1, 1, feat_lengths) 

        return features


########################################### model ###########################################
class DSEE_Retrieval(nn.Module):
    def __init__(self,   
                 config,             
                 med_config = 'configs/med_config.json',  
                 image_size = 224,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                         
                 num_classes=11003,
                 ):              
        super().__init__()
        self.args = config 
        self.num_classes = num_classes
        self.logit_scale = torch.ones([]) * (1 / 0.01)
        self.temp = torch.ones([]) * (1 / 0.01)
        self.embed_dim = 256

        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, self.embed_dim)
        self.text_proj = nn.Linear(text_width, self.embed_dim)

        self.itm_head = nn.Linear(text_width, 2) 

        if self.args.get('id', True):
            self.img_classifier = nn.Linear(self.embed_dim, self.num_classes)
            self.txt_classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.img_classifier.weight.data, std=0.001)
            nn.init.constant_(self.img_classifier.bias.data, val=0.0)
            nn.init.normal_(self.txt_classifier.weight.data, std=0.001)
            nn.init.constant_(self.txt_classifier.bias.data, val=0.0)

        if self.args.get('sm', True):
            self.sm_fc = nn.Linear(self.embed_dim, self.embed_dim)
            self.sm_norm = nn.LayerNorm(self.embed_dim)

        if self.args.get('ts', True):
            num_img_tokens = (image_size // 16) ** 2 + 1     
            num_txt_tokens = self.args.get('max_length', 77)
            self.gate_v_w1 = nn.Linear(num_img_tokens, num_img_tokens, bias=False)
            self.gate_v_w2 = nn.Linear(num_img_tokens, num_img_tokens, bias=False)
            self.gate_t_w1 = nn.Linear(num_txt_tokens, num_txt_tokens, bias=False)
            self.gate_t_w2 = nn.Linear(num_txt_tokens, num_txt_tokens, bias=False)
            self.visul_emb_layer = VisualEmbeddingLayer(input_dim=vision_width,
                                                        embed_dim=self.embed_dim,
                                                        ratio=self.args.get('select_ratio', 0.3))
            self.texual_emb_layer = TexualEmbeddingLayer(input_dim=text_width,
                                                         embed_dim=self.embed_dim,
                                                         ratio=self.args.get('select_ratio', 0.3))

      
    def forward(self, image, caption, idx, generate_captions=None, img_candidates=None, txt_candidates=None, w=0.5, infer = False):
        ret = dict()   
        # image     # bs*3*224*224
        if self.args.get('unsupervised', False):
            # image
            image_embeds, attn_i = self.visual_encoder(image,get_last_attn_map=True)   # b*197*768
            attn_i = attn_i.mean(dim=1)   
            # print(attn_i.shape)
            image_feats = self.vision_proj(image_embeds[:,0,:])  
            image_feats_norm = F.normalize(image_feats,dim=-1)
            #text
            un_text = self.tokenizer(generate_captions, 
                                padding='max_length', 
                                truncation=True, 
                                max_length=self.args.get('max_length', 77), 
                                return_tensors="pt").to(image.device) 
            un_text_output = self.text_encoder(un_text.input_ids, 
                                        attention_mask = un_text.attention_mask,                      
                                        return_dict = True,
                                        mode = 'text',
                                        output_attentions=True)
            un_text_embeds = un_text_output.last_hidden_state       
            un_text_feats = self.text_proj(un_text_embeds[:,0,:])    
            un_text_feats_norm = F.normalize(un_text_feats,dim=-1) 
            if self.args.get('ts', True):
                un_gene_attn_t = un_text_output.attentions[-1].mean(dim=1)  
                un_i_tse_f, un_t_tse_f = self.gated_enhance(
                    image_embeds, attn_i, un_text_embeds, un_text.input_ids, un_gene_attn_t)

                ge_loss = self.oriitc_loss(un_i_tse_f, un_t_tse_f, world=True)
                ret.update({'loss_ge':ge_loss * self.args.get('loss_cap_weight', 1.0)})  

            loss_names = self.args.get('loss_names', ['itc', 'itm'])
            if 'itc' in loss_names:
                itc_loss = self.oriitc_loss(image_feats_norm, un_text_feats_norm, world=True)
                ret.update({'loss_itc': itc_loss * self.args.get('loss_itc_weight', 1.0)})

            if 'itm' in loss_names:
                itm_loss = self.itm_loss(image_embeds, un_text, image_feats_norm, un_text_feats_norm, idx)
                ret.update({'loss_itm': itm_loss * self.args.get('loss_itm_weight', 1.0)})

        else:
            image_embeds, attn_i = self.visual_encoder(image,get_last_attn_map=True)   # b*197*768
            attn_i = attn_i.mean(dim=1) 
            image_feats = self.vision_proj(image_embeds[:,0,:])  
            image_feats_norm = F.normalize(image_feats,dim=-1)
        
            #text
            text = self.tokenizer(caption, 
                              padding='max_length', 
                              truncation=True, 
                              max_length=self.args.get('max_length', 77), 
                              return_tensors="pt").to(image.device) 
            text_output = self.text_encoder(text.input_ids, 
                                        attention_mask = text.attention_mask,                      
                                        return_dict = True,
                                        mode = 'text',
                                        output_attentions=True)
            text_embeds = text_output.last_hidden_state      
            text_feats = self.text_proj(text_embeds[:,0,:])     #bs*256
            text_feats_norm = F.normalize(text_feats,dim=-1)    #bs*256

        
            loss_names = self.args.get('loss_names', ['itc', 'itm'])
            if 'itc' in loss_names:
                itc_loss = self.oriitc_loss(image_feats_norm, text_feats_norm, world=True)
                ret.update({'loss_itc': itc_loss * self.args.get('loss_itc_weight', 1.0)})

            if 'itm' in loss_names:
                itm_loss = self.itm_loss(image_embeds, text, image_feats_norm, text_feats_norm, idx)
                ret.update({'loss_itm': itm_loss * self.args.get('loss_itm_weight', 1.0)})

            # modality-specific id loss (L_id, Eq.21)
            if self.args.get('id', True):
                id_loss = self.id_loss(image_feats_norm, text_feats_norm, idx)
                ret.update({'loss_id': id_loss * self.args.get('loss_id_weight', 1.0)})

            if self.args.get('sm', True):
                alpha = self.args.get('alpha', 1.0)

                if self.args.get('txt_sm', True):
                    img_candidate_feats = []
                    for img in img_candidates:
                        img = img.to(image.device)
                        with torch.no_grad():
                            img_candidate_embeds = self.visual_encoder(img)
                            img_candidate_feats.append(F.normalize(
                                self.vision_proj(img_candidate_embeds[:,0,:]), dim=-1))

                    S_t = torch.stack([text_feats_norm * f for f in img_candidate_feats],
                                      dim=0).mean(dim=0)
                    radius_txt = torch.exp(self.sm_norm(self.sm_fc(S_t)))          # (bs, d)
                    eps_t = torch.normal(mean=0.0, std=alpha, size=radius_txt.shape,
                                         device=radius_txt.device)
                    txt_feats_sm = text_feats_norm + eps_t * radius_txt            # Eq.7

                if self.args.get('img_sm', True):
                    txt_candidate_feats = []
                    for txt in txt_candidates:
                        with torch.no_grad():
                            txt_tokens = self.tokenizer(txt,
                                                padding='max_length',
                                                truncation=True,
                                                max_length=self.args.get('max_length', 77),
                                                return_tensors="pt").to(image.device)
                            txt_output = self.text_encoder(txt_tokens.input_ids,
                                                            attention_mask = txt_tokens.attention_mask,
                                                            return_dict = True,
                                                            mode = 'text')
                            txt_embeds = txt_output.last_hidden_state         # bs*77*768
                            txt_candidate_feats.append(F.normalize(
                                self.text_proj(txt_embeds[:,0,:]), dim=-1))

                    S_v = torch.stack([image_feats_norm * f for f in txt_candidate_feats],
                                      dim=0).mean(dim=0)
                    radius_img = torch.exp(self.sm_norm(self.sm_fc(S_v)))          # (bs, d)
                    eps_v = torch.normal(mean=0.0, std=alpha, size=radius_img.shape,
                                         device=radius_img.device)
                    img_feats_sm = image_feats_norm + eps_v * radius_img          # Eq.8

                if self.args.get('img_sm', True) and self.args.get('txt_sm', True):
                    proxy_loss = self.oriitc_loss(img_feats_sm, txt_feats_sm, world=True)
                elif self.args.get('img_sm', True):
                    proxy_loss = self.oriitc_loss(img_feats_sm, text_feats_norm, world=True)
                elif self.args.get('txt_sm', True):
                    proxy_loss = self.oriitc_loss(image_feats_norm, txt_feats_sm, world=True)
                ret.update({'loss_proxy': proxy_loss * self.args.get('loss_itc_weight', 1.0)})

            if self.args.get('captioning', True):
                gene_text = self.tokenizer(generate_captions, 
                                       padding='max_length', 
                                       truncation=True, 
                                       max_length=self.args.get('max_length', 77), 
                                       return_tensors="pt").to(image.device) 
                gene_text_output = self.text_encoder(gene_text.input_ids, 
                                                 attention_mask = gene_text.attention_mask,                      
                                                 return_dict = True, 
                                                 mode = 'text',
                                                 output_attentions=True)
                gene_text_embeds = gene_text_output.last_hidden_state

                if self.args.get('ts', True):
                    gene_attn_t = gene_text_output.attentions[-1].mean(dim=1)     # bs*m*m
                    # gated fusion of intra- and cross-modal attention (Eq.11-17)
                    i_tse_f, t_tse_f = self.gated_enhance(
                        image_embeds, attn_i, gene_text_embeds, gene_text.input_ids, gene_attn_t)

                    ge_loss = self.oriitc_loss(i_tse_f, t_tse_f, world=True)
                    ret.update({'loss_ge':ge_loss * self.args.get('loss_cap_weight', 1.0)})      
            
        return ret  

    def gated_enhance(self, image_embeds, attn_i, text_embeds, text_input_ids, text_attn):
        bs = image_embeds.size(0)

        a_im_v = attn_i[:, 0, :]                  
        a_im_t = text_attn[:, 0, :]                  

        v_local = F.normalize(self.vision_proj(image_embeds), dim=-1)   # (B, n, d)
        t_local = F.normalize(self.text_proj(text_embeds), dim=-1)      # (B, m, d)
        v_cls = v_local[:, 0, :]                                         # (B, d)
        t_cls = t_local[:, 0, :]                                         # (B, d)

        cm_v = (v_local @ t_cls.unsqueeze(-1)).squeeze(-1)   # (B, n)
        cm_v[:, 0] = -1e4                                
        a_cm_v = torch.softmax(cm_v, dim=-1)                

        cm_t = (t_local @ v_cls.unsqueeze(-1)).squeeze(-1)   # (B, m)
        pad_mask = (text_input_ids != 0)
        cm_t = cm_t.masked_fill(~pad_mask, -1e4)             
        cm_t[:, 0] = -1e4                                  
        cm_t[torch.arange(bs, device=cm_t.device), pad_mask.sum(1) - 1] = -1e4 
        a_cm_t = torch.softmax(cm_t, dim=-1)                 

        a_gated_v = torch.sigmoid(self.gate_v_w1(a_im_v) + self.gate_v_w2(a_cm_v))
        a_gated_t = torch.sigmoid(self.gate_t_w1(a_im_t) + self.gate_t_w2(a_cm_t))

        v_s = self.visul_emb_layer(image_embeds, a_gated_v)
        t_s = self.texual_emb_layer(text_embeds, text_input_ids, a_gated_t)
        return F.normalize(v_s, dim=-1), F.normalize(t_s, dim=-1)
        

    
    ###============== Image-text Contrastive Learning ===================###
    def oriitc_loss(self, image_feature, text_feature, world=False):
        """
        image-text contrastive (ITC) loss, InfoNCE
        """
        if world:
            image_features = all_gather_with_grad(image_feature)
            text_features = all_gather_with_grad(text_feature)
        
        batch_size = image_features.shape[0]
        labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
        labels = labels.to(image_features.device)

    
        # normalized features
        image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_image = self.logit_scale * image_norm @ text_norm.t()    
        logits_per_text = logits_per_image.t()

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t =F.cross_entropy(logits_per_text, labels)
        loss = (loss_i +  loss_t)/2

        return loss
  
    ###============== Image-text Matching ===================###
    def itm_loss(self, image_embeds, text, image_feat, text_feat, idx):
        if self.args.get('unsupervised', False):
            bs = image_embeds.shape[0]
            idx = torch.arange(1, bs + 1, dtype=torch.int64, device=image_embeds.device).view(-1, 1)
            num_gpus = torch.cuda.device_count()
            padding_zeros = torch.zeros((num_gpus - 1) * bs, dtype=torch.int64, device=image_embeds.device).view(-1, 1)
            idxs = torch.cat([idx, padding_zeros], dim=0)
            # print(num_gpus)
            # print(idx.shape)
            # print(idxs.shape)
        else:    
            idx = idx.view(-1, 1)  # 16
            idxs = concat_all_gather(idx)  # 64
        # print(idx.shape)
        # print(idxs.shape)

        encoder_input_ids = text.input_ids.clone()  # bs*seq_len
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        # b*197
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

        # forward the positve image-text pair
        bs = image_embeds.shape[0]
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       )
        with torch.no_grad():
            mask = torch.eq(idx, idxs.t())  # bs*4bs
            # print(mask.shape)

            image_feat_world = concat_all_gather(image_feat)
            text_feat_world = concat_all_gather(text_feat)

            sim_i2t = image_feat @ text_feat_world.t() * self.temp
            sim_t2i = text_feat @ image_feat_world.t() * self.temp

            weights_i2t = F.softmax(sim_i2t, dim=1)
            weights_i2t.masked_fill_(mask, 0) 

            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_t2i.masked_fill_(mask, 0)

        image_embeds_world = all_gather_with_grad(image_embeds)

        image_embeds_neg = []
        for b in range(bs): 
            neg_idx = torch.argmax(weights_t2i[b])  # max
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text (from all ranks) for each image
        input_ids_world = concat_all_gather(encoder_input_ids)
        att_mask_world = concat_all_gather(text.attention_mask)

        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.argmax(weights_i2t[b])
            text_ids_neg.append(input_ids_world[neg_idx])  
            text_atts_neg.append(att_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)  # 2bs*77                + -
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)  # 2bs*197*768           - +
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)  # 2bs*197

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask=text_atts_all,
                                       encoder_hidden_states=image_embeds_all,
                                       encoder_attention_mask=image_atts_all,
                                       return_dict=True,
                                       )
  
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)  # 48*2
        # (1111.....000000)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image_embeds.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        return loss_itm

    def id_loss(self, i_feats, t_feats, labels):
        """
        Modality-specific identity classification loss (Sec. III-D):
        separate classifiers control the distances between different identities
        within each modality.
        """
        image_logits = self.img_classifier(i_feats)
        text_logits = self.txt_classifier(t_feats)

        criterion = nn.CrossEntropyLoss(reduction="mean")

        loss = criterion(image_logits, labels) + criterion(text_logits, labels)

        return loss / 2


def dsee_retrieval(config, pretrained='',**kwargs):
    print(pretrained)
    model = DSEE_Retrieval(config, **kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        # print("missing keys:")
        # print(msg.missing_keys)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      


class GatherLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):

    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
