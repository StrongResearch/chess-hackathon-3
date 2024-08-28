import torch
import torch.nn as nn
import math
from utils.transformer_utils import PositionalEncoding, softmax
from utils.constants import PGN_CHARS

class Model(nn.Module):
    """Transformer Model by PyTorch"""

    def __init__(self, nlayers=10, embed_dim=512, nhead=8, dropout=0.5, device='cpu'):
        super().__init__()
        self.vocab = PGN_CHARS
        self.device = device
        self.embedder = nn.Embedding(len(self.vocab), embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=nlayers)
        self.decoder = nn.Linear(embed_dim, len(self.vocab))
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embedder.weight, -initrange, initrange)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, pgn):
        return [self.vocab.index(c) for c in pgn]
    
    def decode(self, tokens):
        return [self.vocab[t] for t in tokens]
    
    def collate(self, batch, truncate_to=1_000):
        seq_lens = torch.tensor([len(seq) for seq in batch])
        max_seq_len = min(truncate_to, seq_lens.max())
        pad_lens = torch.clamp(max_seq_len - seq_lens, min=0)
        seqs = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)[:,:truncate_to]
        pad_from = max_seq_len - pad_lens
        pad_mask = (pad_from.unsqueeze(1) <= torch.arange(seqs.shape[1]))
        return seqs, pad_mask

    def forward(self, pgn_batch): # pgn_batch: list of pgn strings of varying length
        # encode and batch pgns, truncating and padding
        encoded_pgns = [torch.tensor(self.encode(pgn)) for pgn in pgn_batch]
        batch, pad_mask = self.collate(encoded_pgns)
        # Autoregressive modelling - targets are inputs shifted one to the left.
        inputs = batch[:, :-1].to(self.device)
        targets = batch[:, 1:].to(self.device)
        target_pad_mask = pad_mask[:, 1:].to(self.device)
        seq_len = inputs.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, dtype=torch.bool, device=self.device)
        inputs = self.embedder(inputs) # (batch, token, embed)
        inputs = self.pos_encoder(inputs) # (batch, token, embed)
        inputs = self.encoder(inputs, mask=causal_mask, is_causal=True) # (batch, token, embed)
        logits = self.decoder(inputs)
        return logits, targets, target_pad_mask
    
    def score(self, pgn, move):
        '''
        pgn: string e.g. "1.e4 a6 2.Bc4 "
        move: string e.g. "a5 "
        '''
        # encode single pgn and proposed move
        encoded_pgn = self.encode(pgn)
        encoded_move = self.encode(move)
        inputs = torch.tensor(encoded_pgn + encoded_move).unsqueeze(0)
        # generate causal mask
        seq_len = inputs.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, dtype=torch.bool, device=self.device)
        # forward through the model
        inputs = self.embedder(inputs) # (batch_size, seq_len, embed_dim)
        inputs = self.pos_encoder(inputs) # (batch_size, seq_len, embed_dim)
        inputs = self.encoder(inputs, mask=causal_mask, is_causal=True) # (batch, token, embed)
        logits = self.decoder(inputs) # (batch, token, vocab)
        logits = logits[0] # batch size of 1 for scoring
        # decode probability for proposed move
        char_probabilities = []
        input_idxs_to_query = range(len(encoded_pgn) - 1, inputs.shape[1] - 1)
        for move_char_idx, inputs_idx in enumerate(input_idxs_to_query):
            move_char = encoded_move[move_char_idx]
            char_prob = softmax(logits[inputs_idx].detach())[move_char]
            char_probabilities.append(char_prob.item())
        # return the mean (?) probability for characters in the sequence
        return math.prod(char_probabilities)