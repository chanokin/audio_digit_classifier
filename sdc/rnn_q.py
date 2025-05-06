import torch
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
from torch.quantization import DeQuantStub, QuantStub

class LSD(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.3):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.linear(x)
        x = self.dequant(x)
        x = self.act(x)
        x = self.dropout(x)
        return x
        # return self.dropout(self.act(self.linear(x)))


class RNN(torch.nn.Module):
    default_lstm_config = {
        'hidden_size': 128,
        'num_layers': 1,
        'dropout': 0.3,
        'bidirectional': False,

    }
    def __init__(self, lstm_config: dict[str, int], linear_sizes: list[int]):
        assert isinstance(lstm_config, dict), "lstm_config must be a dictionary"
        assert 'input_size' in lstm_config, "lstm_config must contain 'input_size'"
        assert isinstance(linear_sizes, list), "linear_sizes must be a list"
        assert len(linear_sizes) > 0, "linear_sizes must contain at least one element"

        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.lstm_config = deepcopy(RNN.default_lstm_config)
        self.lstm_config.update(lstm_config)
        self.linear_sizes = [self.lstm_config['hidden_size']] + linear_sizes


        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_config['input_size'],
            hidden_size=self.lstm_config['hidden_size'],
            num_layers=self.lstm_config['num_layers'],
            dropout=self.lstm_config['dropout'],
            bidirectional=self.lstm_config['bidirectional'],
            batch_first=True,
        )

        self.n_classes = linear_sizes[-1]

        self.lstm_out_size = lstm_config['hidden_size'] * (2 if self.lstm_config['bidirectional'] else 1)

        self.linear_layers = []
        for i in range(len(self.linear_sizes) - 1):
            if i == len(self.linear_sizes) - 2:
                # the last layer is a linear layer without activation
                layer = LSD(self.linear_sizes[i], self.linear_sizes[i + 1], dropout=0.0)
            else:
                layer = LSD(self.linear_sizes[i], self.linear_sizes[i + 1])

            self.linear_layers.append(layer)
            self.add_module(self._linear_name(i), layer)

    def _linear_name(self, i):
        return f'linear_{i:03d}'

    def forward(self, x):
        # x: (batch_size, seq_len, mfcc_size, frames)
        x = pad_sequence(x, batch_first=True)
        x = self.quant(x)
        x, _ = self.lstm(x)
        x = self.dequant(x)

        x = x[:, -1, :] # take the last output of the LSTM
        for i in range(len(self.linear_layers)):
            x = getattr(self, self._linear_name(i))(x)

        return x