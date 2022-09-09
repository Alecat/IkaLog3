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

from audioop import bias
import cv2
import numpy as np
import blend_modes
import time
from ikalog.utils import matcher

from ikalog.utils.image_filters.filters import MM_COLOR_BY_HUE
from ikalog.scenes.v3.game.inklings.transform import transform_inklings

_left_list = [1, 65, 130, 195]

"""
Parse information
"""
def is_alive(img):
    _, threshold = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    mask_black = matcher.MM_BLACK()
    black = np.sum(mask_black(threshold) > 1)
    black_ratio = black / (threshold.shape[0]*threshold.shape[1])
    return black_ratio < 0.6



def is_special(img):
    _, threshold = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    mask_white = matcher.MM_WHITE()
    white = np.sum(mask_white(threshold) > 1)
    white_ratio = white / (threshold.shape[0]*threshold.shape[1])
    return white_ratio > 0.8
"""
Entry extraction
"""


def extract_entry(img_entry, team_index=None, player_index=None):
    bias_x = 0
    if team_index == 0:
        bias_x = 5
    elif team_index == 1:
        bias_x = -5
    check_status = img_entry[58:65, 21+bias_x:41+bias_x, :]
    return {
        'full': img_entry,
        # apply a threshold on a small segment of the icon that should exclude the weapon and other UI elements
        'check_status': check_status,
        # 'img_kill_or_assist': img_entry[0:17, 0:20, :],
        # 'img_special': img_entry[0:17, 40:40+20, :],
        # 'img_weapon': img_entry[20:20+37, 4:55, :],
        # 'img_button': img_entry[55:55+26, 0:26, :],
        # 'img_special_charge': img_entry[58:58+22, 30:30+33, :],
    }


def extract_players_image(img_team, team_index=None):
    players = []

    for i, left in enumerate(_left_list):
        img_entry = img_team[:, left:left+65, :]
        e = extract_entry(img_entry, team_index, i)
        players.append(e)
    return players


def extract_players(frame, context):
    img_teams = transform_inklings(frame, context)
    if img_teams['scenario'] == 'none':
        return []
    # cv2.imwrite('pimg/team1.png', img_teams['team1'])
    # cv2.imwrite('pimg/team2.png', img_teams['team2'])

    players_team1 = extract_players_image(img_teams['team1'], 0)
    players_team2 = extract_players_image(img_teams['team2'], 1)

    i = 0
    for player in players_team1:
        player['team'] = 0
        player['alive'] = is_alive(player['check_status'])
        player['special'] = is_special(player['check_status'])
        player['index'] = players_team1.index(player)
        i = i + 1

    for player in players_team2:
        player['team'] = 1
        player['alive'] = is_alive(player['check_status'])
        player['special'] = is_special(player['check_status'])
        player['index'] = 7 - players_team2.index(player)
        i = i + 1

    players = []
    players.extend(players_team1)
    players.extend(players_team2[::-1])

    return players


if __name__ == '__main__':
    import time
    import sys

    img = cv2.imread(sys.argv[1], 1)
    context = {
    }
    t = time.time()
    r = extract_players(img, context)

    i = 0
    print("v3")
    for player in r:
        print( player['team'], 'alive:', player['alive'], 'special:', player['special'])
        # if i == 0:
        for k in [
                'full',
                # 'img_weapon',
                # 'img_button',
                # 'img_special_charge',
                # 'img_kill_or_assist',
                # 'img_special'
                ]:
                cv2.imwrite('pimg/inklings.player%d.%s%s.png' %
                            (player['index'], 'splatted.' if player['alive'] else '', k), player[k])

        # for k in ['weapon', 'kill_or_assist', 'special', 'score']:
        #     cv2.imwrite('scoreboard.player%d.%s.%s.png' %
        #                 (i, k, t), player['img_%s' % k])
        i = i + 1

    print("--- %s seconds ---" % (time.time() - t))
