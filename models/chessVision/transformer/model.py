import io
import torch
import torch.nn as nn
import chess.pgn
from chess import Board
from collections import OrderedDict
from utils.transformer_utils import PositionalEncoding, TransformerEncoderBlock
from utils.data_utils import encode_board
from utils.constants import PIECE_CHARS

class Model(nn.Module):
    """Transformer Model"""
    def __init__(self, nlayers=10, embed_dim=512, nhead=8, head_dim=64, ff_dim=2048, dropout=0.1, causal=True, norm_first=False, ghost=False, device='cpu'):
        super().__init__()
        self.vocab = PIECE_CHARS
        self.device = device
        self.embedder = nn.Embedding(len(self.vocab), embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        enc_params = {"embed_dim": embed_dim, "nhead": nhead, "head_dim": head_dim, "ff_dim": ff_dim, "dropout":  dropout, "causal": causal, "norm_first": norm_first, "ghost": ghost, "device": device}
        layers = OrderedDict([(f"EncoderLayer{i}", TransformerEncoderBlock(**enc_params)) for i in range(nlayers)])
        self.encoder = nn.Sequential(layers)
        self.reducer = nn.Linear(embed_dim, 1)
        self.decoder = nn.Linear(64, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.embedder.weight, -1.0, 1.0)
        nn.init.xavier_uniform_(self.reducer.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.reducer.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, boards): # boards: tensor of boards (batch, 8, 8)
        boards = boards.flatten(1) # (batch, 64)
        boards = self.embedder(boards) # (batch, 64, embed)
        boards = self.pos_encoder(boards) # (batch, 64, embed)
        boards = self.encoder(boards) # (batch, 64, embed)
        boards = self.reducer(boards).squeeze() # (batch, 64)
        boards = self.decoder(boards).squeeze() # (batch)
        return boards.squeeze()

    def score(self, pgn, move):
        '''
        pgn: string e.g. "1.e4 a6 2.Bc4 "
        move: string e.g. "a5 "
        '''
        # init a game and board
        game = chess.pgn.read_game(io.StringIO(pgn))
        board = Board()
        # catch board up on game to present
        for past_move in list(game.mainline_moves()):
            board.push(past_move)
        # push the move to score
        board.push_san(move)
        # convert to tensor, unsqueezing a dummy batch dimension
        board_tensor = torch.tensor(encode_board(board)).unsqueeze(0)
        return self.forward(board_tensor).item()
