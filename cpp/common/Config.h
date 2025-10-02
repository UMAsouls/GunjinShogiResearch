#ifndef CONFIG_H
#define CONFIG_H

#include <tuple>
#include <vector>
#include <map>
#include <string>

#include "cpp/common/Piece.h"

class Config {
    private:
        std::pair<int, int> boardShape;

        int goalHeight;
        std::vector<int> goalPos;
        
        int entryHeight;
        std::vector<int> entryPos;

        int pieceLimit;

        std::map<int, Piece> pieceMap;

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

        Piece getPiece(int id) const; 


};

#endif