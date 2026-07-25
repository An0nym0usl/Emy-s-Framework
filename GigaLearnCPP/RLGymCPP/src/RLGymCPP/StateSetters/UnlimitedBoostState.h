#pragma once
#include "StateSetter.h"

namespace RLGC {
    class UnlimitedBoostState : public StateSetter {
    public:
        bool
            randBallSpeed, randCarSpeed, carsOnGround;
        UnlimitedBoostState(bool randBallSpeed = true, bool randCarSpeed = true, bool carsOnGround = false) :
            randBallSpeed(randBallSpeed), randCarSpeed(randCarSpeed), carsOnGround(carsOnGround) {
        }
        virtual void ResetArena(Arena* arena) override;
    };
}