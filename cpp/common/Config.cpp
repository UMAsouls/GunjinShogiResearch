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

    pieceMap[General] = json["General"];
    pieceMap[LieutenantGeneral] = json["LieutenantGeneral"];
    pieceMap[MajorGeneral] = json["MajorGeneral"];
    pieceMap[Colonel] = json["Colonel"];
    pieceMap[LieutenantColonel] = json["LieutenantColonel"];
    pieceMap[Major] = json["Major"];
    pieceMap[Captain] = json["Captain"];
    pieceMap[FirstLieutenant] = json["FirstLieutenant"];
    pieceMap[SecondLieutenant] = json["SecondLieutenant"];
    pieceMap[Plane] = json["Plane"];
    pieceMap[Tank] = json["Tank"];
    pieceMap[Cavalry] = json["Cavalry"];
    pieceMap[Engineer] = json["Engineer"];
    pieceMap[Spy] = json["Spy"];
    pieceMap[LandMine] = json["LandMine"];
    pieceMap[Frag] = json["Frag"];
    pieceMap[Wall] = json["Wall"];
    pieceMap[Entry] = json["Entry"];
    pieceMap[Enemy] = json["Enemy"];
    pieceMap[Space] = json["Space"];

    eraseMap[EraseFrag::BEF] = json["EraseFrag"]["BEF"];
    eraseMap[EraseFrag::AFT] = json["EraseFrag"]["AFT"];
    eraseMap[EraseFrag::BOTH] = json["EraseFrag"]["BOTH"];
    eraseMap[EraseFrag::NONE] = json["EraseFrag"]["NONE"];

}