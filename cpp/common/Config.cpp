#include <fstream>
#include "json.hpp"

#include "cpp/common/Config.h"
#include "cpp/common/Piece.h"
#include "cpp/common/EraseFrag.h"


void Config::loadFromJson(const std::string& filePath) {
    // JSONファイルを読み込み、各メンバ変数に設定するロジックを実装
    // 例: boardShape, goalHeight, goalPos, entryHeight, entryPos, pieceLimit, pieceMap
    std::ifstream ifs(filePath);
    nlohmann::json json;
    ifs >> json;

    boardShape = { json["BOARD"]["SHAPE"][0], json["BOARD"]["SHAPE"][1] };
    goalHeight = json["BOARD"]["GOAL"]["HEIGHT"];
    goalPos = json["BOARD"]["GOAL"]["POS"].get<std::vector<int>>();
    entryHeight = json["BOARD"]["ENTRY"]["HEIGHT"];
    entryPos = json["BOARD"]["ENTRY"]["POS"].get<std::vector<int>>();
    pieceLimit = json["BOARD"]["PIECE_LIMIT"];

    revPieceMap[General] = json["General"];
    revPieceMap[LieutenantGeneral] = json["LieutenantGeneral"];
    revPieceMap[MajorGeneral] = json["MajorGeneral"];
    revPieceMap[Colonel] = json["Colonel"];
    revPieceMap[LieutenantColonel] = json["LieutenantColonel"];
    revPieceMap[Major] = json["Major"];
    revPieceMap[Captain] = json["Captain"];
    revPieceMap[FirstLieutenant] = json["FirstLieutenant"];
    revPieceMap[SecondLieutenant] = json["SecondLieutenant"];
    revPieceMap[Plane] = json["Plane"];
    revPieceMap[Tank] = json["Tank"];
    revPieceMap[Cavalry] = json["Cavalry"];
    revPieceMap[Engineer] = json["Engineer"];
    revPieceMap[Spy] = json["Spy"];
    revPieceMap[LandMine] = json["LandMine"];
    revPieceMap[Frag] = json["Frag"];
    revPieceMap[Wall] = json["Wall"];
    revPieceMap[Entry] = json["Entry"];
    revPieceMap[Enemy] = json["Enemy"];
    revPieceMap[Space] = json["Space"];

    for(const auto& item : revPieceMap) {
        Piece p = item.first;
        int i = item.second;
        pieceMap[i] = p;
    }

    revEraseMap[EraseFrag::BEF] = json["EraseFrag"]["BEF"];
    revEraseMap[EraseFrag::AFT] = json["EraseFrag"]["AFT"];
    revEraseMap[EraseFrag::BOTH] = json["EraseFrag"]["BOTH"];
    revEraseMap[EraseFrag::NONE] = json["EraseFrag"]["NONE"];

    for(const auto& item : revEraseMap) {
        EraseFrag f = item.first;
        int i = item.second;
        eraseMap[i] = f;
    }

}