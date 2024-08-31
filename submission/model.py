from hackathon_train.train import VisionTransformerTwoHeads
from torch import nn

import io
import torch
import torch.nn as nn
import chess.pgn
import numpy as np
import chess
from collections import OrderedDict, Counter
from models.minimal_lczero.policy_index import policy_index


def board_to_leela_input(board, expanded=True):
    '''
    board: chess.Board
    returns: tensor of shape (1, 19, 8, 8)
    '''
    flipped = False
    if board.turn == chess.BLACK:
        board = board.mirror()
        flipped = True
        
    our_pieces = board.occupied_co[chess.WHITE]
    their_pieces = board.occupied_co[chess.BLACK]

    full_board = (1 << 64) - 1

    planes = np.array([
        board.pawns & our_pieces,
        board.knights & our_pieces,
        board.bishops & our_pieces,
        board.rooks & our_pieces,
        board.queens & our_pieces,
        board.kings & our_pieces,
        board.pawns & their_pieces,
        board.knights & their_pieces,
        board.bishops & their_pieces,
        board.rooks & their_pieces,
        board.queens & their_pieces,
        board.kings & their_pieces,
        board.has_queenside_castling_rights(chess.WHITE) * full_board,
        board.has_kingside_castling_rights(chess.WHITE) * full_board,
        board.has_queenside_castling_rights(chess.BLACK) * full_board,
        board.has_kingside_castling_rights(chess.BLACK) * full_board,
        flipped  * full_board,
        0,
        1 * full_board,
    ], dtype="uint64")

    expanded_planes = np.unpackbits(planes.view("uint8")).reshape(1, 19, 8, 8)

    return expanded_planes if expanded else planes


def leela_policy_to_uci_moves(policy):
    return dict(zip(policy_index, policy))


class Model(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = VisionTransformerTwoHeads(
            in_channels=model['in_channels'],
            patch_size=model['patch_size'],
            embed_dim=model['embed_dim'],
            num_heads=model['num_heads'],
            num_layers=model['num_layers'],
            num_classes1=model['num_classes1'],
            num_classes2=model['num_classes2']
        )
        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.MSELoss()

        self.board = None
        self.pgn = None
        self.policy = None

    def forward(self, x):
        return self.model(x)


    def score(self, pgn, move):
        '''
        pgn: string e.g. "1.e4 a6 2.Bc4 "
        move: string e.g. "a5 "
        '''

        if pgn != self.pgn:
            with torch.no_grad():
                self.pgn = pgn
                game = chess.pgn.read_game(io.StringIO(pgn))
                self.board = chess.Board()
                # catch board up on game to present
                for past_move in list(game.mainline_moves()):
                    self.board.push(past_move)
                
                tensor_input = torch.from_numpy(
                    board_to_leela_input(self.board).astype("float32")
                )
                q, policy_output = self.model.forward(tensor_input)
                policy = policy_output.numpy().flatten()
                self.policy = leela_policy_to_uci_moves(policy)

                print("Analyzed new board")
                print(self.board.unicode())
                print("Q: ", q)
                print("Best moves:", Counter(self.policy).most_common(5))
        
        uci_move = self.board.parse_san(move).uci()

        if uci_move not in self.policy:
            print("Very strange!  Can't recognize move: ", move, uci_move)
        result = float(self.policy.get(uci_move, -1.0))
        return result