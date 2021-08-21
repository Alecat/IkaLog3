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

import cv2
import numpy as np

from ikalog.utils.image_filters.filters import MM_COLOR_BY_HUE
from ikalog.scenes.v2.result.scoreboard.transform import transform_scoreboard

_top_list = [62, 119, 176, 233]

"""
Arrow Detection
"""


def find_my_entry_index(all_players):
    def loss_func(p): return np.sum(image_filters.MM_DARK()(p['img_selected']))
    losses = list(map(loss_func, all_players))
    return np.argmin(losses)


def find_my_entry(all_players):
    return all_players[find_my_entry_index(all_players)]


"""
Entry extraction
"""


def extract_entry(img_entry):
    hh = int(img_entry.shape[0] * 0.55)
    return {
        'full': img_entry,
        'img_selected': img_entry[:, 40:40 + 30, :],
        'img_player': img_entry[:, 86:86 + 41, :],
        'img_weapon': img_entry[:, 175: 175 + 45, :],
        'img_name': img_entry[int(hh/2):int(hh/2)+25, 215: 215 + 150, :],
        'img_score': img_entry[:, 322:322 + 100, :],
        'img_kill_or_assist': img_entry[hh:, 459: 459 + 20, :],
        'img_special': img_entry[hh:, 500: 500 + 20, :],
    }


def extract_players_image(img_team):
    players = []

    for top in _top_list:
        img_entry = img_team[top: top + 50, :, :]
        e = extract_entry(img_entry)
        players.append(e)
    return players


def is_selected(img_selected):
    f = MM_COLOR_BY_HUE(hue=(33 - 5, 33 + 5), visibility=(230, 255))
    score = np.sum(f(img_selected)) / 255
    return score > 200


def extract_players(frame):
    img_teams = transform_scoreboard(frame)

    players_win = extract_players_image(img_teams['win'])
    players_lose = extract_players_image(img_teams['lose'])

    for player in players_win:
        player['team'] = 0
        player['index'] = players_win.index(player)
        player['myself'] = is_selected(player['img_selected'])

    for player in players_lose:
        player['team'] = 1
        player['index'] = players_lose.index(player) + 4
        player['myself'] = is_selected(player['img_selected'])

    players = []
    players.extend(players_win)
    players.extend(players_lose)

    return players


if __name__ == '__main__':
    import time
    import sys

    img = cv2.imread(sys.argv[1], 1)
    r = extract_players(img)

    t = time.time()
    i = 0
    for player in r:
        if player['myself']:
            cv2.imwrite('player%d.full.%s.png' %
                    (i, t), player['full'])
            # cv2.imwrite('player%d.name.%s.png' %
            #         (i, t), player['img_name'])
        # if i == 0:
        #     for k in [
        #         'full',
        #         'img_selected',
        #         'img_player',
        #         'img_weapon',
        #         'img_name',
        #         'img_score',
        #         'img_kill_or_assist',
        #         'img_special']:
        #         cv2.imwrite('scoreboard.player%d.%s.png' %
        #                     (i, k), player[k])

        # for k in ['weapon', 'kill_or_assist', 'special', 'score']:
        #     cv2.imwrite('scoreboard.player%d.%s.%s.png' %
        #                 (i, k, t), player['img_%s' % k])
        i = i + 1
