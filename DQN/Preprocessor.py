class Preprocessor:
    """状態の正規化を行うクラス"""
    def __init__(self, scr_width, scr_height):
        self.scr_width = scr_width
        self.scr_height = scr_height
        # _2: 1/2, _4: 1/4, _8: 1/8
        scr_h_2 = scr_height / 2
        scr_h_4 = scr_h_2 / 2
        scr_h_8 = scr_h_4 / 2
        scr_w_2 = scr_width / 2
        scr_w_4 = scr_w_2 / 2

        # 状態正規化用の平均と標準偏差を指定
        self.norm_param = {
            'player_y' : (scr_h_2, scr_h_8),
            'player_vel' : (0, 0.5),
            'player_dist_to_ceil' : (scr_h_4, scr_h_8),
            'player_dist_to_floor' : (scr_h_4, scr_h_8),
            'next_gate_dist_to_player' : (scr_w_2, scr_w_4),
            'next_gate_block_top' : (scr_h_2, scr_h_8),
            'next_gate_block_bottom' : (5.5 * scr_h_8, scr_h_8)
        }

    def normalize(self, state):
        for k in state.keys():
            state[k] -= self.norm_param[k][0]
            state[k] /= self.norm_param[k][1]
        return state