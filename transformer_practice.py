"""
- 문제 출처: 데이콘 태양광 데이터 셋
    - 선정이유: 데이터 셋 크기가 엄청 작아서 학습속도 높아서 피드백 바로 알기 쉬움 & 베이스라인 코드(Tree)가 있어서 비교하기 쉬움
- 문제: 1400일 정도의 학습셋에 대해서 7일 데이터로 학습하고 다음날 2일에 대한 TARGET 값을 맞추는 문제
"""

import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import math

from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.activation import MultiheadAttention
from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.preprocessing import StandardScaler
from datetime import datetime



class PositionalEncoding(nn.Module):
    """
    Pytorch transformer 튜토리얼에서 그대로 가지고 옴
    """
    def __init__(self, multi_head_concat_hidden_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, multi_head_concat_hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, multi_head_concat_hidden_dim, 2).float() * (-math.log(10000.0) / multi_head_concat_hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, input_feat_dim, multi_head_concat_hidden_dim, nhead, npf_hid, nlayers, dropout=0.3):
        """
        - input_feat_dim: # feature (multivariate )
        - multi_head_concat_hidden_dim: (hidden_dim per head) X (num head)
        - npf_hid: pointwise feedforward dim
        """

        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(multi_head_concat_hidden_dim, dropout)

        self.encoder_embedding = nn.Linear(input_feat_dim, multi_head_concat_hidden_dim)
        encoder_layers = TransformerEncoderLayer(
            d_model=multi_head_concat_hidden_dim, nhead=nhead, dim_feedforward=npf_hid, dropout=dropout, activation='relu'
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=nlayers)

        # TARGET dimension이 1이기 때문에 1 X multi_head_concat_hidden_dim
        self.decoder_embedding = nn.Linear(1, multi_head_concat_hidden_dim)
        decoder_layers = TransformerDecoderLayer(
            d_model=multi_head_concat_hidden_dim, nhead=nhead, dim_feedforward=npf_hid, dropout=dropout, activation='relu'
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layers, num_layers=nlayers)
        self.fc = nn.Linear(multi_head_concat_hidden_dim, 1)

        self.multi_head_concat_hidden_dim = multi_head_concat_hidden_dim

    def generate_square_subsequent_mask(self, seq_len):
        mask = torch.tril(torch.ones(seq_len, seq_len)) == 1
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask, tgt, tgt_mask):
        src = self.encoder_embedding(src) * math.sqrt(self.multi_head_concat_hidden_dim)
        # batch first --> sequence first로 바꿔주기
        src = src.transpose(1, 0)
        # src = self.pos_encoder(src)   # 여기서는 안해도 될듯... = input의 모든 sequence를 다 보고 QKV연산 진행하겠다

        tgt = self.decoder_embedding(tgt) * math.sqrt(self.multi_head_concat_hidden_dim)
        # batch first --> sequence first로 바꿔주기
        tgt = tgt.transpose(1, 0)
        tgt = self.pos_encoder(tgt)  # pos_encoder 안에서 dropout 함

        memory = self.transformer_encoder(src, src_mask)
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=None)

        output = output.transpose(0, 1)
        output = self.fc(output)

        return output


def preprocess(df):
    """
    간단한 전처리 -> time으로 변환해서 cutoff time 표시하기 등
    """
    df['time'] = (df['Day'] * 24*60 + df['Hour']*60 + df['Minute']) * 60
    df['time'] = df['time'].apply(lambda x: datetime.fromtimestamp(x)) - pd.Timedelta(hours=9)

    y_list = []
    for start_i in range(7, df['Day'].max(), 2):
        cond1 = df['Day'] == start_i
        cond2 = df['Day'] == start_i + 1
        y = df.loc[cond1 | cond2, ["time", "TARGET"]].reset_index(drop=True)
        y = y.rename(columns={"time": "target_time"})
        y['cutoff_time'] = y.loc[0]['target_time']
        y_list.append(y)

    if y_list:
        y = pd.concat(y_list)
    else:
        y = None

    df.drop(["Day"], axis=1, inplace=True)
    return df, y



if __name__ == "__main__":
    FEATURES = ["DHI", "DNI", "WS", "RH", "T"]
    TARGET = "TARGET"
    device = "cuda"
    multi_head_concat_hidden_dim = 40  # embedding dimension
    npf_hid = 40  # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 4  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # the number of heads in the multiheadattention models
    dropout = 0.1  # the dropout value
    lr = 0.03  # learning rate
    assert multi_head_concat_hidden_dim % nhead == 0

    model = TransformerModel(
        len(FEATURES), multi_head_concat_hidden_dim, nhead, npf_hid, nlayers, dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    criterion = nn.MSELoss()


    train_df = pd.read_csv("train/train.csv")
    train_df, y = preprocess(train_df)

    """
    데이터를 아래와 같이 나눔
             X                y
    --------------------   --------
            7days            2days
    --------------------   --------

    (2day 씩 shift하면서 생성)
    """
    x_list = []
    y_list = []
    for cutoff_time in y['cutoff_time'].unique():
        cond1 = train_df['time'] < cutoff_time
        cond2 = train_df['time'] >= cutoff_time - pd.Timedelta(days=7)
        x = train_df.loc[cond1 & cond2, FEATURES]
        _y = y.loc[y['cutoff_time'] == cutoff_time, TARGET]

        x_list.append(x)
        y_list.append(_y)


    scaler = StandardScaler()
    scaler.fit(pd.concat(x_list))
    for i, x in enumerate(x_list):
        x_list[i] = scaler.transform(x)

    # Tensorize (Batch first shape)
    tensor_x = torch.FloatTensor(np.dstack(x_list).transpose(2, 0, 1))
    tensor_y = torch.FloatTensor(np.stack(y_list)).unsqueeze(2)

    dataset = TensorDataset(tensor_x, tensor_y)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

    model.train()

    for epoch in range(500):
        total_loss = 0.
        for batch in data_loader:
            optimizer.zero_grad()

            x_, y_ = batch
            # Make sequence first --> Transformer foward() 안에서 진행함
            x_ = x_.to(device)
            y_ = y_.to(device)

            #
            # sequence 길이에 맞게 mask 계산
            #
            if x_.shape[1] != 48 * 7:   # X feature: 하루당 데이터 48개 X 7일
                src_mask = model.generate_square_subsequent_mask(x_.shape[1]).to(device)
            else:
                src_mask = model.generate_square_subsequent_mask(48 * 7).to(device)

            if y_.shape[1] != 48 * 2:     # y target: 하루당 데이터 48개 X 2일
                tgt_mask = model.generate_square_subsequent_mask(y_.shape[1]).to(device)
            else:
                tgt_mask = model.generate_square_subsequent_mask(48 * 2).to(device)

            output = model(x_, src_mask, y_, tgt_mask)

            loss = criterion(output, y_)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(total_loss / len(data_loader))
