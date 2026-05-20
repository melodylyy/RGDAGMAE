import math
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from dgl.nn import GraphConv, SAGEConv, ChebConv
from dataload import Embedding
from dgl.nn.pytorch import utils
from torch_geometric.nn import GINConv
from dgl.nn.pytorch.conv import SAGEConv

import dgl
import torch as th
import torch.nn.functional as F
from torch.nn import Module

class GraphMAE(nn.Module):
    def __init__(self, args):
        super(GraphMAE, self).__init__()

        # Define Graph Convolutional layers (encoder)
        self.conv1 = SAGEConv(args.d_h, args.d_h, aggregator_type=args.aggregator_type)
        self.conv2 = SAGEConv(args.d_h, args.d_h, aggregator_type=args.aggregator_type)

        # Gating layers to control feature flow after graph convolution
        self.gate1 = nn.Linear(args.d_h, args.d_h)
        self.gate2 = nn.Linear(args.d_h, args.d_h)

        # Activation function
        self.relu = nn.ReLU()

        # Define decoder to reconstruct features from latent representations
        self.decoder = nn.Linear(args.d_h, args.d_h)

        # Mask ratio for feature masking
        self.mask_ratio = args.mask_ratio

    def mask_features(self, features):
        """
        Randomly masks a portion of the features.
        Returns the masked features and the mask index.
        """
        num_nodes, feature_dim = features.size()
        mask = th.rand(num_nodes, feature_dim) < self.mask_ratio  # Masking matrix
        masked_features = features.clone()
        masked_features[mask] = 0  # Masked features set to 0
        return masked_features, mask

    def scaled_cosine_error(self, reconstructed_feat, original_feat, mask):
        """
        Compute the scaled cosine error between the reconstructed features and the original features.
        Only computes for the masked (missing) features.
        """
        # Normalize the vectors
        reconstructed_feat = F.normalize(reconstructed_feat, p=2, dim=1)
        original_feat = F.normalize(original_feat, p=2, dim=1)

        # Cosine similarity (dot product of normalized vectors)
        cosine_sim = th.sum(reconstructed_feat * original_feat, dim=1)  # Shape: [num_nodes]

        # Ensure mask is 1D (node-level mask)
        if mask.dim() > 1:  # If the mask is 2D, reduce to 1D by max across features
            mask = mask.max(dim=1)[0]  # Take max across feature dimension to get a node-level mask

        # Ensure the mask has the same shape as cosine_sim (1D, num_nodes)
        if mask.shape[0] != cosine_sim.shape[0]:
            raise ValueError(f"Mask size mismatch: {mask.shape[0]} vs {cosine_sim.shape[0]}")

        # Apply the mask to select the relevant cosine similarities
        cosine_sim_masked = cosine_sim[mask]  # Mask is now correctly applied to the cosine similarity

        # Scaling factor (optional: can be adjusted based on difficulty or distance)
        scaling_factor = th.norm(reconstructed_feat, p=2, dim=1)[mask]  # Ensure mask is 1D
        scaling_factor = th.clamp(scaling_factor, min=1e-6)  # Avoid division by zero

        # Compute scaled cosine error
        scaled_cosine_error = 1 - cosine_sim_masked / scaling_factor  # Higher error for harder-to-reconstruct features

        return th.mean(scaled_cosine_error)  # Return the mean error

    def forward(self, g, in_feat):
        # Mask features
        masked_feat, mask = self.mask_features(in_feat)


        # Encoder: Compute latent representations Z
        z = self.conv1(g, masked_feat)  # First SAGEConv layer
        gate1_output = th.sigmoid(self.gate1(z))  # Gate 1
        z = z * gate1_output  # Apply gating mechanism

        z = self.relu(z)  # Apply activation
        z = self.conv2(g, z)  # Second SAGEConv layer
        gate2_output = th.sigmoid(self.gate2(z))  # Gate 2
        z = z * gate2_output  # Apply gating mechanism

        # Decoder: Reconstruct features from latent representations Z
        reconstructed_feat = self.decoder(z)

        # Compute the scaled cosine error loss (only for masked values)
        cosine_loss = self.scaled_cosine_error(reconstructed_feat, in_feat, mask)

        return z, cosine_loss



class GCNConnation(nn.Module):
    def __init__(self):
        super(GCNConnation, self).__init__()

    def forward(self, train_g, h):
        train_g.ndata['h'] = h

        e = train_g.edges(form='all')

        src = e[0]

        dst = e[1]

        src_embed = train_g.nodes[src].data['h']

        dst_embed = train_g.nodes[dst].data['h']

        memory = th.cat((src_embed, dst_embed), 1)


        return memory




def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()

    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)


    return pad_attn_mask.expand(batch_size, len_q, len_k)
class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        self.args = args

    def forward(self, Q, K, V):
        # Scaled dot-product attention
        scores = th.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.args.d_h)
        attn = nn.Softmax(dim=-1)(scores)
        context = th.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.args = args

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(args.d_h, args.d_h * args.n_heads, bias=False)
        self.W_K = nn.Linear(args.d_h, args.d_h * args.n_heads, bias=False)
        self.W_V = nn.Linear(args.d_h, args.d_h * args.n_heads, bias=False)

        # Final linear projection
        self.fc = nn.Linear(args.n_heads * args.d_h, args.d_h, bias=False)

        # Learnable parameters for Agent Attention
        self.A = nn.Parameter(th.randn(args.d_h, args.d_h))  # Shared projection matrix
        self.layer_norm = nn.LayerNorm(args.d_h)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        # Move inputs to the correct device
        input_Q = input_Q.to(self.args.device)
        input_K = input_K.to(self.args.device)
        input_V = input_V.to(self.args.device)

        residual, batch_size = input_Q, input_Q.size(0)

        # Linear projections and reshape for multi-head attention
        Q = self.W_Q(input_Q).view(batch_size, -1, self.args.n_heads, self.args.d_h).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.args.n_heads, self.args.d_h).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.args.n_heads, self.args.d_h).transpose(1, 2)

        # Step 1: Compute VV using Softmax Attention (A^T K)
        A_T = self.A.transpose(0, 1)  # Transpose of A
        scores_VV = th.matmul(A_T.unsqueeze(0).unsqueeze(0), K.transpose(-1, -2)) / np.sqrt(self.args.d_h)
        attn_VV = nn.Softmax(dim=-1)(scores_VV)
        VV = th.matmul(attn_VV, V)  # Intermediate representation VV

        # Step 2: Compute final context using Linear Attention (Q A VV)
        Q_A = th.matmul(Q, self.A.unsqueeze(0).unsqueeze(0))  # Q A
        context = th.matmul(Q_A, VV.transpose(-1, -2))  # Q A VV

        # Reshape and project the final output
        context = context.transpose(1, 2).reshape(batch_size, -1, self.args.n_heads * self.args.d_h)
        outputs = self.fc(context)

        # Add residual connection and apply layer normalization
        outputs = self.layer_norm(outputs + residual)

        return outputs, attn_VV


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super(EncoderLayer, self).__init__()
        self.args = args
        self.enc_self_attn = MultiHeadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)
        self.layer_norm = nn.LayerNorm(args.d_h)

    def forward(self, enc_inputs, enc_self_attn_mask):
      
        device = enc_inputs.device
        enc_self_attn_mask = enc_self_attn_mask.to(device)

        # Multi-Head Attention
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

    
        #print("enc_outputs device:", enc_outputs.device)

        enc_inputs = enc_inputs.to(self.args.device)
        #print("enc_inputs device:", enc_inputs.device)

        
        enc_outputs = self.layer_norm(enc_outputs + enc_inputs)

        # Feed-Forward Network
        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn



class DecoderLayer(nn.Module):
    def __init__(self,args):
        super(DecoderLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.d_h * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        outputs = self.fc(inputs)
        # Note: In a full decoder implementation, you would typically have a self-attention block here too.
        return outputs


# The Encoder and Decoder classes remain largely unchanged, but ensure they align with the modified layers above.

class Encoder(nn.Module):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.enc_layers)])

    def forward(self, enc_inputs,embedding):
        enc_outputs = get_token_embedding(enc_inputs, embedding)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        #print("esamesam",enc_self_attn_mask.shape)

        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)

        enc_outputs = enc_outputs.view(enc_outputs.size()[0], -1)
        return enc_outputs


class Decoder(nn.Module):
    def __init__(self,args):
        super(Decoder, self).__init__()
        self.predict = DecoderLayer(args)

    def forward(self, enc_output):
        pred = self.predict(enc_output)
        return pred





class PoswiseFeedForwardNet(nn.Module):

    def __init__(self,args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args = args
        self.layer_norm = nn.LayerNorm(args.d_h)
        self.fc = nn.Sequential(
            nn.Linear(args.d_h, args.n_heads * args.d_h),
            nn.ReLU(),
            nn.Linear(args.n_heads * args.d_h, args.d_h)
        )



    def forward(self, inputs):
        residual = inputs
        outputs = self.fc(inputs)
        return self.layer_norm(outputs + residual)


def get_token_embedding(datax, embedding):
    token_embedding = []
    for pair in datax:
        embed = []
     
        embed.append(embedding[pair[0]].detach().cpu().numpy())
        embed.append(embedding[pair[1] + 431].detach().cpu().numpy())
        token_embedding.append(embed)

    
    #token_embed = th.Tensor(token_embedding)
    # token_embed = th.tensor(token_embedding, dtype=th.float32)
    token_embedding_np = np.array(token_embedding, dtype=np.float32)  
    token_embed = th.from_numpy(token_embedding_np)

    #print(f"token_embedding size: {len(token_embedding)}, type: {type(token_embedding)}")

    return token_embed


np.random.seed(1)
attn = np.random.rand()


class Connection(nn.Module):
    def __init__(self):
        super(Connection, self).__init__()
        self.attn = th.nn.Parameter(th.tensor(attn), requires_grad=True).float()


    def forward(self, outputs1, outputs2):

        outputs = self.attn * outputs1 + (1 - self.attn) * outputs2
        return outputs


class Transformer(nn.Module):
    def __init__(self,args):
        super(Transformer, self).__init__()
        self.encoder = Encoder(args)
        self.encoder2 = GCNConnation()
        self.decoder = Decoder(args)
        self.con = Connection()

    def forward(self, enc_inputs, train_g, feature, embedding):
        outputs1 = self.encoder(enc_inputs,embedding)
        outputs2 = self.encoder2(train_g, feature)
        outputs = self.con(outputs1, outputs2)
        pred = self.decoder(outputs)
        return pred

    def test(self, test_x, test_g, h):
        test_g.ndata['h'] = h
        e = test_g.edges(form='all')
        src = e[0]
        dst = e[1]
        src_embed = test_g.nodes[src].data['h']
        dst_embed = test_g.nodes[dst].data['h']
        memory2 = th.cat((src_embed, dst_embed), 1)
        memory1 = self.encoder(test_x)
        memory = self.con(memory1, memory2)
        pred = self.decoder(memory)
        return pred



class LYY_model(nn.Module):
    def __init__(self, args):
        super(LYY_model, self).__init__()
        self.graphsage = GraphMAE(args)
        self.transformer = Transformer(args)
        self.decoder = Decoder(args)
        self.con = Connection()

    def construct_graph(self, id, m_embed, d_embed):
     
        src = id[:, 0]
        dst = id[:, 1] + m_embed.size(0) 
        num_nodes = m_embed.size(0) + d_embed.size(0)
        # print("drug_idx", num_nodes)
        # print("iiiiiiiiiid", id)
        # print("src",src.shape)
        # print("dst", dst.shape)

        
        graph = dgl.graph((src, dst), num_nodes=num_nodes)

        node_features = th.cat((m_embed, d_embed), dim=0)
        #print("node_features", node_features.shape)

  
        graph.ndata['feature'] = node_features
        return graph, node_features

    def forward(self, id, m_embed, d_embed):
        graph,embedding = self.construct_graph(id, m_embed, d_embed)
        g_feature,mae_loss = self.graphsage(graph, graph.ndata['feature'])
        pred_score = self.transformer(id, graph, g_feature,embedding)
        return pred_score,mae_loss






























