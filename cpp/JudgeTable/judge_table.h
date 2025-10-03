#include <map>

#include "cpp/common/Config.h"
#include "cpp/common/Piece.h"
#include "cpp/common/Player.h"
#include "cpp/common/JudgeFrag.h"

//駒の勝敗判定テーブル管理クラス
class JudgeTable {
private:
    std::map<Piece, std::map<Piece, JudgeFrag>> table_p1;
    std::map<Piece, std::map<Piece, JudgeFrag>> table_p2;

    std::map<Player, std::map<Piece, std::map<Piece, JudgeFrag>>> tables;

    Config config;

    void TableSet(std::map<Piece, std::map<Piece, JudgeFrag>>& table);

public:
    JudgeTable(const Config& config);
    ~JudgeTable() {};

    void FragSet(Player player, Piece p);
    JudgeFrag GetJudge(Player player, Piece p1, Piece p2) { return tables[player][p1][p2]; }

};