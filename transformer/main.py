# -*- coding: utf-8 -*-
# @Time : 2023/7/20 11:56
import copy
import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab):
        """
        :param d_model:word embedding维度
        :param vocab:词表中词的数量
        """
        super(Embeddings, self).__init__()
        #  Embeddings继承自nn.Module，子类如果用父类的init方法需要用super方法，该方法可以解决多重继承的父类查找问题
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        :param x:一个batch的输入，size = [batch, L] L为句子最长长度
        :return:
        """
        return self.lut(x) * math.sqrt(self.model)  # 乘了一个权重，不改变维度 size=[B,L,d_model]
        #  为什么要乘一个常数？因为需要放大，为了位置向量相加的时候不会因为位置向量大而忽视


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=500):
        """
        位置编码类
        :param d_model: 位置编码维度，一般与word embedding 相同，方便相加
        :param dropout: 用于缓解过拟合的方法，随机将神经元的激活值置为0
        :param max_len: 语料中的最长句子的长度，即word embedding的L
        """
        super(PositionalEncoding, self).__init__()
        #  定义dropout,神经元以p的几率不被激活，一般用在全连接神经网络映射层之后，训练用测试需关闭
        self.dropout = nn.Dropout(p=dropout)

        #  计算position encoding
        pe = torch.zeros(max_len,d_model) # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0,max_len).unsqueeze(1) #  建立词的位置以便公式计算，size=[max_len,1]
        div_term = torch.exp(torch.arange(0,d_model, 2) *- (math.log(10000.0) / d_model)) # 计算公式中的10000 ** (2i/d_model)
        pe[:, 0::2] = torch.sin(position * div_term) # 计算偶数维度的pe
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数
        pe = pe.unsqueeze(0) # size=(1,L,d_model) # 为了后续与word embedding相加，意味batch维度下的操作相同
        # 虽然是绝对位置计算方式，但体现了相对位置信息
        self.register_buffer("pe",pe) #pe值不参加训练

    def forward(self, x):
        #  输入的最终编码 = word embedding + positional embedding
        x = x +  Variable(self.pe[:, :x.size(1)], requires_grad=False) # size=[batch,l,d_model]
        return self.dropout(x) # size不变

def attention(query,key,value,mask=None,dropout=None):
    """
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k) #  矩阵乘法，
    if mask is not None:
        scores = scores.masked_fill(mask=mask,value=torch.tensor(-1e9))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn,value),p_attn

def clones(module, N):
    #  定义N个相同的模块
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self,h,d_model,dropout):
        """
        实现多头注意力
        :param h: 头数
        :param d_model: word embedding维度
        :param dropout:
        """
        super(MultiHeadAttention,self).__init__()
        assert d_model % h == 0 # 检查word embedding 维度能否被h整除
        self.d_k = d_model // h
        self.h = h
        self.liners = clones(nn.Linear(d_model,d_model),4) #四个线性变换
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask= None):
        """
        :param query:输入x=word_emb+pos_emb, size=[batch,l,d_model]
        :param key:
        :param value:
        :param mask:掩码矩阵，编码器mask的size=[batch,l,src_l]，解码器mask的size=[batch,tag_l,tgt_l]
        :return:
        """
        if mask  is not None:
            mask = mask.unsqueeze(1) # 编码器mask_size=[batch,1,1,src_L],解码器mask_size=[batch,1,tag_l,tag_l]
        nbatches = query.size(0)  # 获取batch的值，nbatches=batch

        # 1) 利用三个全连接全出QKV的向量，再维度变换[batch,L,d_model]->[batch,h,l,d_model/h]
        query, key,value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
            for l, x in zip(self.liners, (query,key,value))
        ]

        # 2) 实现Scaled Dot-Product  Attention
        x,self.attn = attention(query,key,value,mask=mask,dropout=self.dropout)
        # 3）实现拼接
        x = x.transpose(1,2).contiguous().view(nbatches,-1, self.h,*self.d_k)

        return self.liners[-1](x)

class Batch:
    def __init__(self,src,trg=None,pad=0):
        """
        :param src: 一个batch的输入，size=[bsz,src_L]
        :param trg: 一个batch的输出，size=[bsz,trg_L]
        :param pad:
        """
        self.src=src
        self.src_mask = (src != pad).unsqueeze(-2) #返回一个true/false的矩阵，szie=[bsz,1,src_L]
        if trg is not None:
            self.trg = trg[:,:-1] #用于输入模型，不带末尾<eos>
            self.trg_y = trg[:,1:] #用于计算损失函数，不带起始<sos>
            self.trg_mask = self.make_std_mask(self.trg,pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod #静态方法，没有self参数，放在类的命名空间内好维护
    def make_std_mask(tgt,pad):
        """
        :param tgt: 一个batch的target,size=[batch,tgt_L]
        :param pad: 用于padding的值，一般为0
        :return: mask, size=[bsz,tgt_L,tgt_L]
        """
        tgt_mask = (tgt != pad).unsqueeze(-2) #返回一个true/false矩阵，size=[bsz,1,tgt_L]
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        # 两个mask求和得到最终的mask [bsz,1,L]&[1,size,size]=[bsz,tgt_L,tgt_L]
        return tgt_mask

def subsequent_mask(size):
    """
    :param size:输出的序列长度
    :return: 返回下三角矩阵，size=[1,size,size]
    """
    attn_shape = (1,size,size)
    subsequent_mask = np.triu(np.ones(attn_shape),1).astype("uint8") #返回上三角矩阵，不带轴线
    return torch.from_numpy(subsequent_mask) == 0 #返回等于0的部分，其实就是下三角矩阵

class LayerNorm(nn.Module):
    def __init__(self, features,eps=1e-6):
        """
        实现层归一化
        :param features:
        :param eps:
        """
        super(LayerNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)) # 对归一化后的结果线性偏移，
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps=eps #保证分母不为0
    def forward(self,x):
        """
        :param x: 输入x,size=[bsz,L,d_model]
        :return: 归一化后的结果，size不变
        """
        mean = x.mean(-1,keepdim=True) # 最后一个维度求均值
        std = x.std(-1,keepdim=True) #最后一个维度求方差
        return self.a_2 * (x-mean) / (std+self.eps) + self.b_2 # 归一化并线性缩放加偏移

class SublayerConnection(nn.Module):
    """
    实现残差连接
    """
    def __init__(self,size,dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size) #初始化贵归一化函数
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,sublayer):
        """
        :param x: 当前子层的输入，size=[bsz,L,d_model]
        :param sublayer:当前子层的前向传播函数，指代多头attention或前馈神经网络
        :return:
        """
        return self.norm(x+self.dropout(sublayer(x)))

#FFN(x) = max(0,xW1+b1)W2 + b2
class PositionwiseFeedForward(nn.Module):
    """实现全连接层"""
    def __init__(self,d_model,d_ff,dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    """
    Encoder层整体封装，由self-attention\残差链接、归一化和前馈神经网络组成
    """
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer,self).__init__()
        self.aelf_attn = self_attn #定义多头注意力，即传入一个多头注意力类
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout),2)
        self.size = size
    def forward(self,x,mask):
        """
        :param x: 输入x,即(word_embedding+positional embedding),size=[bsz,l,d_model]
        :param mask:掩码矩阵，编码器的mask的size=[bsz,1,src_l]
        :return:size=[bsz,l,d_model]
        """
        x = self.sublayer[0](x,lambda x:self.aelf_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward) #实现前馈和残差链接，size=[bsz,l,d_model]

class Encoder(nn.Module):
    """
    Encoder最终封装，由若干个Encoder Layer组成
    """
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
    def forwrard(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super(DecoderLayer,self).__init__()
        self.size=size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size,dropout),3)

    def forward(self,x,memory,src_mask,tgt_mask):
        """
        :param x:target,size=[bsz,tgt_L,d_model]
        :param memory:encoder的输出，size=[bsz,src_L,d_model]
        :param src_mask:源数据的mask,size=[bsz,1,src_l]
        :param tgt_mask:标签的mask,size = [bsz,tgt_L,tgt_l]
        :return:
        """
        m = memory
        x = self.sublayer[0](x, lambda x:self.self_attn(x,x,x,tgt_mask))
        x = self.sublayer[1](x,lambda x:self.src_attn(x,m,m,src_mask))
        return self.sublayer[2](x,self.feed_forward)

class Decoder(nn.Module):
    """解码器的高层封装，由N个Decoder layer组成"""
    def __init__(self,layer,N):
        super(Decoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = LayerNorm(layer.size)
    def forward(self,x,memory,src_mask,tgt_mask):
        for layer in self.layers:
            x = layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)


class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self,src,tgt,src_mask,tgt_mask):
        return self.decoder(self.encoder(src,src_mask),src_mask,tgt,tgt_mask)

    def encoder(self,src,src_mask):
        return self.encoder(self.src_embed(src),src_mask)
    def decoder(self,memory,src_mask,tgt,tgt_mask):
        return self.decoder(self.tgt_embed(tgt),memory,src_mask,tgt_mask)


class Generator(nn.Module):
    """
    定义一个全连接层+softmax
    """
    def __init__(self,d_model,vocab):
        super(Generator,self).__init__()
        self.proj = nn.Linear(d_model,vocab)

    def forward(self,x):
        return F.softmax(self.proj(x),dim=-1)