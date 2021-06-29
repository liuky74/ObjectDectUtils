import torch
from torch.nn import functional as F
from torch import nn
import math

"""
self attention自关注模块思想:
将input分为QKV三个部分
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, in_ch, out_ch, head_num=1):
        super(MultiHeadAttention, self).__init__()
        self.linear_q = nn.Linear(in_ch, out_ch)
        self.linear_k = nn.Linear(in_ch, out_ch)
        self.linear_v = nn.Linear(in_ch, out_ch)
        self.final_linear = nn.Linear(out_ch, out_ch)
        self.head_num = head_num
        self.head_dim = out_ch // head_num
        assert out_ch % head_num == 0

    def self_attention(self, query, key, value, mask=None, dropout=None):
        """
        将src+pos的数据相乘,如果query和key来自于同一个输入x,得到的就是x中每个像素信息之间的关系矩阵
        如果query和key来自不同的输入x y,则得到的是x和y各个像素之间的关系矩阵"""
        head_dim = query.size(-1)
        # 将key转置后与query矩阵乘,因为query==key==src+pos,因此等价于src中每个像素特征的点乘和(思考10个像素,每个像素的特征维度为5,
        # 即矩阵[10,5]与自身转置[5,10]的矩阵乘,即5个特征维度依次点乘后求和,最终得到矩阵[10,10]即每个像素与其他像素的特征点乘和)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)  # 矩阵乘后激活
        if dropout is not None:
            p_attn = dropout(p_attn)
        """
        将像素相关性矩阵与value矩阵乘,即将像素之间相关性加权至value矩阵上,
        value的shape表示为[h*w,feature_channel_num],关系矩阵attn的shape表示为:[h*w,h*w],attn[0][1]表示像素0与像素1之间的相关性,
        根据矩阵乘法的计算特性,attn*value即表示将attn中保存的各个像素的相关性权重加权至对应的像素上,并且是每个channel都加权一次,
        最终得到经过相关性加权的value值"""
        ret = torch.matmul(p_attn, value)
        return ret, p_attn

    def forward(self, x: torch.Tensor,y:torch.Tensor):
        query = self.linear_q(x)  # 加权形成x的像素特征,为什么是像素特征,因为胡扯的,黑盒算法,作者BB是什么就是什么
        key = self.linear_k(y)  # 加权形成y的索引特征,同上,胡扯的
        value = self.linear_v(x)  #  加权x
        # 多头并行
        batch_size = x.shape[0]

        query = query.view(batch_size, -1, self.head_num, self.head_dim)
        query = torch.transpose(query, 1, 2)  # 形成shape:[batch_size,head_num,h*w,head_dim]的数据

        key = key.view(batch_size, -1, self.head_num, self.head_dim)
        key = torch.transpose(key, 1, 2)

        value = value.view(batch_size, -1, self.head_num, self.head_dim)
        value = torch.transpose(value, 1, 2)

        # 自注意力函数
        value, att_mat = self.self_attention(query, key, value)
        value = torch.transpose(value, 1, 2)  # 重新形成[batch,h*w,head_num,head_dim]的数据
        res = self.final_linear(value)
        return res


if __name__ == '__main__':
    multi_head_att = MultiHeadAttention(256, 256, 8)
    x = torch.randn(2, 28 * 28, 256)
    res = multi_head_att(x)

    # Q,K,V shape: [batch_size,28*38,256]
    # query = key = value = torch.randn(2,28*38,256)
    #
    #
    # linear_q=nn.Linear(256,256)
    # linear_k=nn.Linear(256,256)
    # linear_v=nn.Linear(256,256)
    # final_linear = nn.Linear(256,256)
    # query = linear_q(query)
    # query = query.view(2,-1,8,32)
    # query = torch.transpose(query,1,2)
    #
    # key = linear_k(key)
    # key = key.view(2,-1,8,32)
    # key = torch.transpose(key,1,2)  # 形成shape:[batch_size,head_num,h*w,head_dim]的数据
    # value = linear_v(value)
    # value = value.view(2,-1,8,32)
    # value = torch.transpose(value,1,2)
    #
    # value,src_att = multi_self_attention(query, key, value) # 返回了添加相关性的src,以及相关性矩阵
    #
    # value = torch.transpose(value,1,2)  # 重新形成[batch,h*w,head_num,head_dim]的数据
    #
    # res = final_linear(value)

print("query")
