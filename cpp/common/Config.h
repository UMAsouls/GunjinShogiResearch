#ifndef CONFIG_H
#define CONFIG_H

#include <tuple>
#include <vector>
#include <map>
#include <string>

#include "cpp/common/Piece.h"
#include "cpp/common/EraseFrag.h"

class Config {
    private:
        std::pair<int, int> boardShape;

        int goalHeight;
        std::vector<int> goalPos;
        
        int entryHeight;
        std::vector<int> entryPos;

        int pieceLimit;

        std::map<int, Piece> pieceMap;
        std::map<int, EraseFrag> eraseMap;
        std::map<Piece, int> revPieceMap;
        std::map<EraseFrag, int> revEraseMap;

        std::vector<Piece> goalPieces;

    public:
        Config() {};
        ~Config() {};

        void loadFromJson(const std::string& filePath);

        std::pair<int, int> getBoardShape() const { return boardShape; }

        int getGoalHeight() const { return goalHeight; }
        std::vector<int> getGoalPos() const { return goalPos; }

        int getEntryHeight() const { return entryHeight; }
        std::vector<int> getEntryPos() const { return entryPos; }

        int getPieceLimit() const { return pieceLimit; }    

        Piece getPiece(int id) const { return pieceMap.at(id); } 
        EraseFrag getEraseFrag(int id) const { return eraseMap.at(id); } 

        int getPieceId(const Piece& p) const { return revPieceMap.at(p); }
        int getEraseFragId(const EraseFrag& f) const { return revEraseMap.at(f); }

        bool isGoalPiece(Piece p) const;

};

#endif