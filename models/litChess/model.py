import io
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


class Model():
    def __init__(self):
        super().__init__()
        self.board = None
        self.pgn = None
        self.policy = None
        
        self.model = None


    def score(self, pgn, move):
        '''
        pgn: string e.g. "1.e4 a6 2.Bc4 "
        move: string e.g. "a5 "
        '''

        if pgn != self.pgn:
            self.pgn = pgn
            game = chess.pgn.read_game(io.StringIO(pgn))
            self.board = chess.Board()
            # catch board up on game to present
            for past_move in list(game.mainline_moves()):
                self.board.push(past_move)
            
            tensor_input = board_to_leela_input(self.board)
            # TODO: plug in model here
            # policy_output, q = model.forward()
            policy_output = np.random.normal(size=(1858))
            q = np.random.normal()
            self.policy = leela_policy_to_uci_moves(policy_output)

            print("Analyzed new board")
            print(self.board.unicode())
            print("Q: ", q)
            print("Best moves:", Counter(self.policy).most_common(5))
        
        uci_move = self.board.parse_san(move).uci()

        if uci_move not in self.policy:
            print("Very strange!  Can't recognize move: ", move, uci_move)
        return self.policy.get(uci_move, -1)