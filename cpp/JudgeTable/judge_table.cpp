

#include "cpp/JudgeTable/judge_table.h"
#include "cpp/common/Piece.h"
#include "cpp/common/JudgeFrag.h"

constexpr Piece PIECE_LIST[] = {
    Space,
    General, LieutenantGeneral, MajorGeneral,
    Colonel, LieutenantColonel, Major,
    Captain, FirstLieutenant, SecondLieutenant,
    Plane, Tank,
    Cavalry, Engineer,
    Spy,
    LandMine, Frag
};

void FillTable(std::map<Piece, JudgeFrag>& m, JudgeFrag fill) {
    for(const auto& p: PIECE_LIST) {
        m[p] = fill;
    }
}

//テーブルの初期設定
void JudgeTable::TableSet(std::map<Piece, std::map<Piece, JudgeFrag>>& table) {
    //大将の設定
    table[General] = std::map<Piece, JudgeFrag>();
    FillTable(table[General], JudgeFrag::WIN);
    table[Piece::General][Piece::Spy] = JudgeFrag::LOSE;
    table[Piece::General][Piece::LandMine] = JudgeFrag::DRAW;

    //中将の設定
    table[LieutenantGeneral] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::LieutenantGeneral], JudgeFrag::WIN);
    table[Piece::LieutenantGeneral][Piece::General] = JudgeFrag::LOSE;
    table[Piece::LieutenantGeneral][Piece::LandMine] = JudgeFrag::DRAW;

    //少将の設定
    table[Piece::MajorGeneral] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::MajorGeneral], JudgeFrag::WIN);
    table[Piece::MajorGeneral][Piece::General] = JudgeFrag::LOSE;
    table[Piece::MajorGeneral][Piece::LieutenantGeneral] = JudgeFrag::LOSE;
    table[Piece::MajorGeneral][Piece::LandMine] = JudgeFrag::DRAW;

    //大佐の設定
    table[Piece::Colonel] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::Colonel], JudgeFrag::WIN);
    table[Piece::Colonel][Piece::General] = JudgeFrag::LOSE;
    table[Piece::Colonel][Piece::LieutenantGeneral] = JudgeFrag::LOSE;
    table[Piece::Colonel][Piece::MajorGeneral] = JudgeFrag::LOSE;
    table[Piece::Colonel][Piece::Plane] = JudgeFrag::LOSE;
    table[Piece::Colonel][Piece::Tank] = JudgeFrag::LOSE;
    table[Piece::Colonel][Piece::LandMine] = JudgeFrag::DRAW;

    //中佐の設定
    table[Piece::LieutenantColonel] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::LieutenantColonel], JudgeFrag::WIN);
    table[Piece::LieutenantColonel][Piece::General] = JudgeFrag::LOSE;
    table[Piece::LieutenantColonel][Piece::LieutenantGeneral] = JudgeFrag::LOSE;
    table[Piece::LieutenantColonel][Piece::MajorGeneral] = JudgeFrag::LOSE;
    table[Piece::LieutenantColonel][Piece::Colonel] = JudgeFrag::LOSE;
    table[Piece::LieutenantColonel][Piece::Plane] = JudgeFrag::LOSE;
    table[Piece::LieutenantColonel][Piece::Tank] = JudgeFrag::LOSE;
    table[Piece::LieutenantColonel][Piece::LandMine] = JudgeFrag::DRAW;

    //少佐の設定
    table[Piece::Major] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::Major], JudgeFrag::WIN);
    table[Piece::Major][Piece::General] = JudgeFrag::LOSE;
    table[Piece::Major][Piece::LieutenantGeneral] = JudgeFrag::LOSE;
    table[Piece::Major][Piece::MajorGeneral] = JudgeFrag::LOSE;
    table[Piece::Major][Piece::Colonel] = JudgeFrag::LOSE;
    table[Piece::Major][Piece::LieutenantColonel] = JudgeFrag::LOSE;
    table[Piece::Major][Piece::Plane] = JudgeFrag::LOSE;
    table[Piece::Major][Piece::Tank] = JudgeFrag::LOSE;
    table[Piece::Major][Piece::LandMine] = JudgeFrag::DRAW;

    //大尉の設定
    table[Piece::Captain] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::Captain], JudgeFrag::LOSE);
    table[Piece::Captain][Piece::FirstLieutenant] = JudgeFrag::WIN;
    table[Piece::Captain][Piece::SecondLieutenant] = JudgeFrag::WIN;
    table[Piece::Captain][Piece::Cavalry] = JudgeFrag::WIN;
    table[Piece::Captain][Piece::Engineer] = JudgeFrag::WIN;
    table[Piece::Captain][Piece::Spy] = JudgeFrag::WIN;

    //中尉の設定
    table[Piece::FirstLieutenant] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::FirstLieutenant], JudgeFrag::LOSE);
    table[Piece::FirstLieutenant][Piece::SecondLieutenant] = JudgeFrag::WIN;
    table[Piece::FirstLieutenant][Piece::Cavalry] = JudgeFrag::WIN;
    table[Piece::FirstLieutenant][Piece::Engineer] = JudgeFrag::WIN;
    table[Piece::FirstLieutenant][Piece::Spy] = JudgeFrag::WIN;

    //少尉の設定
    table[Piece::SecondLieutenant] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::SecondLieutenant], JudgeFrag::LOSE);
    table[Piece::SecondLieutenant][Piece::Cavalry] = JudgeFrag::WIN;
    table[Piece::SecondLieutenant][Piece::Engineer] = JudgeFrag::WIN;
    table[Piece::SecondLieutenant][Piece::Spy] = JudgeFrag::WIN;

    //騎兵の設定
    table[Piece::Cavalry] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::Cavalry], JudgeFrag::LOSE);
    table[Piece::Cavalry][Piece::Engineer] = JudgeFrag::WIN;
    table[Piece::Cavalry][Piece::Spy] = JudgeFrag::WIN;

    //工兵の設定
    table[Piece::Engineer] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::Engineer], JudgeFrag::LOSE);
    table[Piece::Engineer][Piece::Spy] = JudgeFrag::WIN;
    table[Piece::Engineer][Piece::LandMine] = JudgeFrag::WIN;

    //ヒコーキの設定
    table[Piece::Plane] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::Plane], JudgeFrag::WIN);
    table[Piece::Plane][Piece::General] = JudgeFrag::LOSE;
    table[Piece::Plane][Piece::LieutenantGeneral] = JudgeFrag::LOSE;
    table[Piece::Plane][Piece::MajorGeneral] = JudgeFrag::LOSE;

    //タンクの設定
    table[Piece::Tank] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::Tank], JudgeFrag::WIN);
    table[Piece::Tank][Piece::General] = JudgeFrag::LOSE;
    table[Piece::Tank][Piece::LieutenantGeneral] = JudgeFrag::LOSE;
    table[Piece::Tank][Piece::MajorGeneral] = JudgeFrag::LOSE;
    table[Piece::Tank][Piece::LandMine] = JudgeFrag::DRAW;

    //スパイの設定
    table[Piece::Spy] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::Spy], JudgeFrag::LOSE);
    table[Piece::Spy][Piece::General] = JudgeFrag::WIN;

    //地雷の設定
    table[Piece::LandMine] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::LandMine], JudgeFrag::LOSE);
    table[Piece::LandMine][Piece::Plane] = JudgeFrag::LOSE;
    table[Piece::LandMine][Piece::Engineer] = JudgeFrag::LOSE;

    //フラグの設定
    //これはこのタイミングでは設定しない
    table[Piece::Frag] = std::map<Piece, JudgeFrag>();
    FillTable(table[Piece::Frag], JudgeFrag::LOSE);

    for(const auto& p: PIECE_LIST) {
        //同じ駒同士は引き分け、空きマスには勝ち
        table[p][p] = JudgeFrag::DRAW;
        table[p][Piece::Space] = JudgeFrag::WIN;
    }
    
}

JudgeTable::JudgeTable(const Config& c): config(c) {
    TableSet(table_p1);
    TableSet(table_p2);

    tables[Player::PLAYER_ONE] = table_p1;
    tables[Player::PLAYER_TWO] = table_p2;
}

void JudgeTable::FragSet(Player player, Piece piece) {
    //フラグの勝敗判定を設定
    tables[player][Piece::Frag] = tables[player][piece];

    //相手側に設定を反映
    Player opponent = (player == Player::PLAYER_ONE) ? Player::PLAYER_TWO : Player::PLAYER_TWO;
    for(const auto& p: PIECE_LIST) {
        tables[opponent][p][Piece::Frag] = tables[player][p][piece];
    }
}