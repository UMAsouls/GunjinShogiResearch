#include "cpp/JudgeBoard/judge_board.h"

#include <tuple>

//アクション作成関数
int make_action(int fromX, int fromY, int toX, int toY, const std::pair<int, int>& board_shape) {
    // 指定された座標に基づいてアクションを作成するロジックを実装
    int size_flatten = board_shape.first * board_shape.second;
    int width = board_shape.first;
    int height = board_shape.second;

    return (fromX * width + fromY) * size_flatten + (toX * width + toY);
}

void get_not_plane_move_in_range(
    int x, int y, const std::pair<int, int>& board_shape, 
    const std::vector<std::vector<Piece>>& crossRange, std::vector<int>& legalMoves,
    int dir, int dist
) {
    int x_dirs[] = {1,-1,0,0};
    int y_dirs[] = {0,0,1,-1};

    int dx = x_dirs[dir];
    int dy = y_dirs[dir];

    for(int i = 0; i < dist; i++) {
        if(crossRange[dir].size() >= i) break;
        if(crossRange[dir][i] != Piece::Space && crossRange[dir][i] != Piece::Enemy) break;

        legalMoves.push_back(make_action(x, y, x+dx*(i+1), y+dy*(i+1), board_shape));
    }
}

//特殊移動をしない駒の判定
void get_normal_piece_moves(
    int x, int y, const std::pair<int, int>& board_shape, 
    const std::vector<std::vector<Piece>>& crossRange, std::vector<int>& legalMoves
) {
    // 移動可能な位置をlegalMovesに追加
    // 右左上下の順番
    for(int i = 0; i < 4; i++) {
        get_not_plane_move_in_range(x,y,board_shape,crossRange,legalMoves,i,1);
    }
}

//騎兵と戦車の判定
void get_tank_cavalry_moves(
    int x, int y, const std::pair<int, int>& board_shape, 
    const std::vector<std::vector<Piece>>& crossRange, std::vector<int>& legalMoves
) {
    // 移動可能な位置をlegalMovesに追加
    // 右
    get_not_plane_move_in_range(x,y,board_shape,crossRange,legalMoves,0,1);
    // 左
    get_not_plane_move_in_range(x,y,board_shape,crossRange,legalMoves,1,1);
    //上だけ2マス
    get_not_plane_move_in_range(x,y,board_shape,crossRange,legalMoves,2,2);
    // 下
    get_not_plane_move_in_range(x,y,board_shape,crossRange,legalMoves,3,1);
}

//工兵の判定
void get_engineer_moves(
    int x, int y, const std::pair<int, int>& board_shape, 
    const std::vector<std::vector<Piece>>& crossRange, std::vector<int>& legalMoves
) {
    // 移動可能な位置をlegalMovesに追加
    int width = board_shape.first;
    int height = board_shape.second;
    //工兵は移動距離無限
    int dist = width ? width > height : height;
    // 右左上下の順番
    for(int i = 0; i < 4; i++) {
        get_not_plane_move_in_range(x,y,board_shape,crossRange,legalMoves,i,dist);
    }
}

//ヒコーキの判定
void get_plane_moves(
    int x, int y, const std::pair<int, int>& board_shape, 
    const std::vector<std::vector<Piece>>& crossRange, std::vector<int>& legalMoves
) {
    int height = board_shape.second;

    //右左
    get_not_plane_move_in_range(x,y,board_shape,crossRange,legalMoves,0,1);
    get_not_plane_move_in_range(x,y,board_shape,crossRange,legalMoves,1,1);

    //上下
    for(int d = 2; d < 4; d++) {
        int dy = 1 ? d == 2 : -1;
        for(int i = 0; i < height; i++) {
            if(crossRange[d].size() >= i) break;
            if(crossRange[d][i] != Piece::Space && crossRange[d][i] != Piece::Enemy) continue;

            legalMoves.push_back(make_action(x, y, x, y+dy*(i+1), board_shape));
        }   
    }
}


void JudgeBoard::reset() {
    board_p1.reset();
    board_p2.reset();
}

void JudgeBoard::erase(int x, int y, Player player) {
    getBoard(player).erase(x, y);
}

void JudgeBoard::move(int fromX, int fromY, int toX, int toY, Player player) {
    getBoard(player).move(fromX, fromY, toX, toY);
}

JudgeFrag JudgeBoard::getJudge(int fromX, int fromY, int toX, int toY, Player player) const {
    Board& board = getBoard(player);
    Board& o_board = getBoard(getOpponent(player));

    auto ref_pos = get_reflect_pos(toX, toY);

    Piece p1 = board.get(fromX, fromY);
    Piece p2 = o_board.get(ref_pos.first, ref_pos.second);

    return judgeTable.GetJudge(player, p1, p2);
}

// 合法手の取得
std::vector<int>& JudgeBoard::getLegalMoves(Player player) const {
    std::vector<int> legalMoves;
    Board board = getBoard(player);

    // ボードから合法手を作成
    std::pair<int,int> shape = config.getBoardShape();
    for(int y = 0; y < shape.second; y++) {
        for(int x = 0; x < shape.first; x++) {
            Piece p = board.get(x,y);
            std::vector<std::vector<Piece>> cross_range = board.getCrossRange(x,y);
            switch (p)
            {
            case Tank:
            case Cavalry:
                get_tank_cavalry_moves(x,y,shape,cross_range,legalMoves);
                break;
            case Engineer:
                get_engineer_moves(x,y,shape,cross_range,legalMoves);
                break;
            case Plane:
                get_plane_moves(x,y,shape,cross_range,legalMoves);
                break;
            case Space:
            case Wall:
            case Entry:
            case Enemy:
            case LandMine:
            case Frag:
                break;
            default:
                get_normal_piece_moves(x,y,shape,cross_range,legalMoves);
                break;
            }
        }
    }

    return legalMoves;
}

bool JudgeBoard::isGameOver(Player player) const {
    Board& board = getBoard(player);

    std::vector<std::pair<int,int>> goal_pos;
    for(int pos: config.getGoalPos()) {
        goal_pos.push_back({pos, config.getGoalHeight()-1});
    }

    for(const auto& pos: goal_pos) {
        if(config.isGoalPiece(board.get(pos.first, pos.second))) {
            return true;
        }
    }

    return false;
}