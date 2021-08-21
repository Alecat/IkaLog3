#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  IkaLog
#  ======
#  Copyright (C) 2017 Takeshi HASEGAWA
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from ikalog.scenes.v2.game.splatzone_tracker import SplatzoneTracker
from .game.in_game import Spl2InGame
from .game.session import Spl2GameSession
from .game.timer_icon import GameTimerIcon
from .game.start import Spl2GameStart

from .game.special_meter import Spl2GameSpecialMeter
#from .game.special_gauge.background import V2GameSpecialGaugeBackground as GameSpecialGaugeBackground
#from .game.special_gauge.gauge import V2GameSpecialGauge as GameSpecialGauge
#from .game.special_gauge.level import V2GameSpecialGaugeLevel as GameSpecialGaugeLevel
#from .game.special_gauge.sub_and_special import V2GameSubAndSpecial as GameSubAndSpecial

from .game.special_weapon_activation import Spl2GameSpecialWeaponActivation
from .game.map import Spl2GameMap
from .game.kill import Spl2GameKill
from ikalog.scenes.game.kill_combo import GameKillCombo
from .game.dead import Spl2GameDead
from .game.low_ink import Spl2GameLowInk
from .game.finish import Spl2GameFinish
from .game.paint_tracker import V2PaintTracker
from .game.objective_tracker import ObjectiveTracker
from .game.splatzone_tracker import SplatzoneTracker
from .game.ranked_battle_events import Spl2GameRankedBattleEvents
from .game.sub_weapon import Spl2GameSubWeapon
from .game.go_sign import GameGoSign
from .game.inklings.simple import Spl2GameInklings
#from .lobby import V2Lobby as Lobby

from .result.judge import Spl2ResultJudge
from .result.map import Spl2ResultMap
from .result.scoreboard.simple import Spl2ResultScoreboard

from .salmon_run.session import Spl2SalmonRunSession
from .salmon_run.game_start import Spl2SalmonRunGameStart
from .salmon_run.weapon_specified import Spl2SalmonRunWeaponSpecified
from .salmon_run.norma import Spl2SalmonRunNorma
from .salmon_run.wave_start import Spl2SalmonRunWaveStart
from .salmon_run.wave_finish import Spl2SalmonRunWaveFinish
from .salmon_run.player_status import Spl2SalmonRunPlayerStatus
from .salmon_run.result import Spl2SalmonRunResultJudge
from .salmon_run.game_over import Spl2SalmonRunGameOver
from .salmon_run.count import Spl2SalmonRunTimeCounter
from .salmon_run.mr_grizz import Spl2SalmonRunMrGrizz


from ikalog.scenes.lobby import Lobby
from ikalog.scenes.blank import Blank
from ikalog.scenes.botw.dead import BOTWDead


def initialize_scenes(engine):
    s = [
        Spl2GameSession(engine),
        Spl2InGame(engine),

        Spl2ResultMap(engine),
        Spl2ResultJudge(engine),
        Spl2ResultScoreboard(engine),

        GameGoSign(engine),
        Spl2GameDead(engine),
        Spl2GameFinish(engine),
        Spl2GameKill(engine),
        GameKillCombo(engine),
        # Spl2GameLowInk(engine),
        Spl2GameMap(engine),
        Spl2GameRankedBattleEvents(engine),
        Spl2GameSpecialMeter(engine),
        Spl2GameSpecialWeaponActivation(engine),
        Spl2GameStart(engine),
        Spl2GameSubWeapon(engine),
        ObjectiveTracker(engine),
        SplatzoneTracker(engine),
        GameTimerIcon(engine),
        Spl2GameInklings(engine),

        # V2PaintTracker(engine),
        Lobby(engine),

        # Spl2SalmonRunSession(engine),
        # Spl2SalmonRunGameStart(engine),
        # Spl2SalmonRunWeaponSpecified(engine),
        # Spl2SalmonRunNorma(engine),
        # Spl2SalmonRunGameOver(engine),
        # Spl2SalmonRunTimeCounter(engine),
        # Spl2SalmonRunResultJudge(engine),
        # Spl2SalmonRunWaveStart(engine),
        # Spl2SalmonRunWaveFinish(engine),
        # Spl2SalmonRunPlayerStatus(engine),
        # Spl2SalmonRunMrGrizz(engine),

        # BOTWDead(engine),

        # Blank(engine),

        Blank(engine),
    ]
    return s
