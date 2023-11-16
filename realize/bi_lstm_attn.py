import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 3
embedding_dim = 2
n_hidden = 5
num_classes = 2  # 0 or 1

# 数据为简单的三个单词即=sequence_length = 3
sentences = ["i love you", "he loves me", "she likes baseball",
             "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 => good 0 => bad

vocab = list(set(" ".join(sentences).split()))
print(f"vocab:{vocab}")
word2idx = {w: i for i, w in enumerate(vocab)}
print(f"word2idx:{word2idx}")
vocab_size = len(word2idx)


def make_data(sentences):
    inputs = []
    for sen in sentences:
        inputs.append(np.asarray([word2idx[n] for n in sen.split()]))

    targets = []
    for out in labels:
        targets.append(out)  # 用softmax做损失函数

    return torch.LongTensor(inputs), torch.LongTensor(targets)


inputs, targets = make_data(sentences)
dataset = Data.TensorDataset(inputs, targets)
loader = Data.DataLoader(dataset, batch_size, True)


class BiLSTM_Attention(nn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=True)
        # bi_lstm => bidirectional = True
        # self.lstm = nn.LSTM(embedding_dim, n_hidden, bidirectional=False)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
    def attention_net(self, lstm_output, final_state):
        batch_size = len(lstm_output)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        hidden = final_state.view(batch_size, -1, 1)
        # torch.bmm() 矩阵乘法
        # squeeze()移除维度 注意力权重=>(batch_size, seq_length)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # softmax函数进行归一化处理
        soft_attn_weights = F.softmax(attn_weights, 1)

        # context : [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(
            1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def forward(self, X):
        '''
        X: [batch_size, seq_len]
        '''
        input = self.embedding(X)
        # input : [batch_size, seq_len, embedding_dim]
        # input : [seq_len, batch_size, embedding_dim]
        input = input.transpose(0, 1)

        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        # output : [batch_size, seq_len, n_hidden]
        output = output.transpose(0, 1)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        # model : [batch_size, num_classes], attention : [batch_size, n_step]
        return self.out(attn_output), attention


if __name__ == '__main__':
    # 模型、优化器
    model = BiLSTM_Attention().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    for epoch in range(500):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred, attention = model(x)
            loss = criterion(pred, y)
            if (epoch + 1) % 1000 == 0:
                print('Epoch:', '%04d' % (epoch + 1),
                      'cost =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test
    test_text = 'i hate me'
    tests = [np.asarray([word2idx[n] for n in test_text.split()])]
    test_batch = torch.LongTensor(tests).to(device)

    # Predict
    predict, _ = model(test_batch)
    predict = predict.data.max(1, keepdim=True)[1]
    if predict[0][0] == 0:
        print(test_text, "is Bad Mean...")
    else:
        print(test_text, "is Good Mean!!")
