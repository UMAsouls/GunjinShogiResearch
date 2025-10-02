#include <fstream>
#include "json.hpp"

#include "cpp/common/Config.h"


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

    for (const auto& item : json["pieceMap"].items()) {
        int id = std::stoi(item.key());
        Piece piece = item.value().get<Piece>();
        pieceMap[id] = piece;
    }

}