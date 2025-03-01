#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  IkaLog
#  ======
#  Copyright (C) 2015 Takeshi HASEGAWA
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
import logging
import sys
import time
import traceback

import cv2
import numpy as np
import copy

from ikalog.scenes.stateful_scene import StatefulScene
from ikalog.utils import *
from ikalog.constants import stages_v2, modes_v3
from ikalog.utils.ikamatcher2.matcher import MultiClassIkaMatcher2 as MultiClassIkaMatcher

from ikalog.ml.classifier import ImageClassifier
from ikalog.ml.text_reader import TextReader


logger = logging.getLogger()

mode_check_coordinates = [
    374, 46, 524, 550
]
mode_check_coordinate_offset = {
    'tricolor': [0, -46]
}


class Spl3GameStart(StatefulScene):

    def reset(self):
        super(Spl3GameStart, self).reset()
        self.stage_votes = []
        self.rule_votes = []
        self.power_votes = []

        self._last_event_msec = - 100 * 1000
        self._last_run_msec = - 100 * 1000

    def elect(self, context, votes):
        # Discard too old data.
        election_start = context['engine']['msec'] - self.election_period
        votes = list(filter(lambda e: election_start < e[0], votes))

        # count
        items = {}
        for vote in votes:
            if vote[1] is None:
                continue
            key = vote[1]
            items[key] = items.get(key, 0) + 1

        # return the best key
        sorted_keys = sorted(
            items.keys(), key=lambda x: items[x], reverse=True)
        sorted_keys.extend([None])  # fallback

        return sorted_keys[0]

    def _detect_stage_and_rule(self, context):
        frame = context['engine']['frame']

        # mask white areas to black
        start_white = cv2.inRange(frame, (240, 240, 240), (255, 255, 255))

        img_test = copy.deepcopy(frame)
        img_test[start_white > 0] = (0, 0, 0)

        # Check for the splatter with the mode name
        found_splatter = False
        if self._mask_start.match(img_test):
            found_splatter = True

        for rule_id in mode_check_coordinate_offset.keys():
            if not found_splatter:
                mode_splatter = img_test[mode_check_coordinates[1]+mode_check_coordinate_offset[rule_id][1]:mode_check_coordinates[1]+mode_check_coordinates[3]+mode_check_coordinate_offset[rule_id][1],
                    mode_check_coordinates[0]+mode_check_coordinate_offset[rule_id][0]:mode_check_coordinates[0]+mode_check_coordinates[2]+mode_check_coordinate_offset[rule_id][0]]
                if self._mask_start.match(mode_splatter):
                    found_splatter = True

        if not found_splatter:
            return None, None

        best_stage = self._stage_masks.match_best(frame)
        best_rule = self._rule_masks.match_best(frame)

        stage, rule = None, None
        if best_stage[1] is not None:
            stage = best_stage[1].id_
        if best_rule[1] is not None:
            rule = best_rule[1].id_
        stage = None

        return stage, rule


    def _read_power_estimation(self, context):
        frame = context['engine']['frame']
        img_power_bgr = frame[652:652+45, 88:88+100, :]
        img_power_gray = matcher.MM_WHITE()(img_power_bgr)
        try:
            n = self._tr.read_int(img_power_gray)
            assert n > 500
            assert n < 5000
            self.power_votes.append((context['engine']['msec'],str(n)))
        except:
            pass


    def _state_default(self, context):
        # pass matching in some scenes.
        # session = self.find_scene_object('Spl2GameSession')
        # if session is not None:
        #     if not (session._state.__name__ in ('_state_default')):
        #         return False

        frame = context['engine']['frame']
        if frame is None:
            return False
        stage, rule = self._detect_stage_and_rule(context)

        matched = (stage or rule)

        if matched:
            self.stage_votes = []
            self.rule_votes = []
            self.stage_votes.append((context['engine']['msec'], stage))
            self.rule_votes.append((context['engine']['msec'], rule))
            self._switch_state(self._state_tracking)
            return True

        return False

    def _state_tracking(self, context):
        frame = context['engine']['frame']

        if frame is None:
            return False

        self._read_splash_tag(context)
        stage, rule = self._detect_stage_and_rule(context)
        matched = (stage or rule)

        # 画面が続いているならそのまま
        if matched:
            self.rule_votes.append((context['engine']['msec'], rule))
            return True

        # 1000ms 以内の非マッチはチャタリングとみなす
        if not matched and self.matched_in(context, 1000):
            # Check for white screen
            if np.mean(frame) > 250:
                self._elect(context)
                self._switch_state(self._state_default)
            return False

        # それ以上マッチングしなかった場合 -> シーンを抜けている
        if not self.matched_in(context, 20000, attr='_last_event_msec'):
            self._elect(context)

        self._switch_state(self._state_default)
        return False

    def _elect(self, context):
        context['game']['map'] = self.elect(context, self.stage_votes)
        context['game']['rule'] = self.elect(context, self.rule_votes)

        if context['game']['rule'] in ('nawabari', 'area', 'yagura', 'hoko', 'asari'):
            power = self.elect(context, self.power_votes)
            if power:
                context['game']['gachi_power'] = int(power)

        if not context['game']['start_time']:
            logger.debug("SETTING FALLBACK START TIME")
            # start_time should be initialized in GameGoSign.
            # This is a fallback in case GameGoSign was skipped.
            context['game']['start_time'] = IkaUtils.getTime(context)
            context['game']['start_offset_msec'] = \
                context['engine']['msec']

        self._call_plugins('on_game_start')
        self._last_event_msec = context['engine']['msec']

    def _read_splash_tag(self, context):
        frame = context['engine']['frame']
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(frame, (3,3), 0)
        edges = cv2.Canny(image=img_blur, threshold1=0, threshold2=150) 

        # cv2.imwrite('pimg/edges.png', edges)
        print("GETTING CONTOURS")
        bg = np.zeros(frame.shape)
        contours, hierachy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        biggest = 0
        splashtag_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # if (area > 10):
            splashtag_contours.append(contour)
        cv2.drawContours(bg, splashtag_contours, -1, (255, 255, 255), 3).astype(bool) 


        # cv2.imwrite('pimg/contours.png', bg)

    def _analyze(self, context):
        pass

    def dump(self, context):
        for v in self.stage_votes:
            if v[1] is None:
                continue
            logger.debug('stage', v[0], v[1])

        for v in self.rule_votes:
            if v[1] is None:
                continue
            logger.debug('rule', v[0], v[1])

    def _init_scene(self, debug=False):
        self.election_period = 5 * 1000  # msec

        self.mapname_left = 1010
        self.mapname_width = 246
        self.mapname_top = 580
        self.mapname_height = 40

        self._rule_masks = MultiClassIkaMatcher()
        self._stage_masks = MultiClassIkaMatcher()

        self._mask_start = IkaMatcher(
            mode_check_coordinates[0],
            mode_check_coordinates[1],
            mode_check_coordinates[2],
            mode_check_coordinates[3],
            img_file='v3_start_mode.png',
            threshold= 0.8,
            orig_threshold= 0.12,
            bg_method=matcher.MM_DARK(visibility=(20, 255)),
            fg_method=matcher.MM_BLACK(),
            label='start/mode',
            call_plugins=self._call_plugins,
            debug=False
        )

        for rule_id in modes_v3.keys():
            try:
                rule = IkaMatcher(
                    460, 270, 360, 220,
                    img_file='v3_mode_%s.png' % rule_id,
                    threshold=0.95,
                    orig_threshold=0.30,
                    bg_method=matcher.MM_BLACK(visibility=(0, 215)),
                    fg_method=matcher.MM_WHITE(visibility=(150,255)),
                    label='rule:%s' % rule_id,
                    call_plugins=self._call_plugins,
                    debug=debug,
                )
                setattr(rule, 'id_', rule_id)
                self._rule_masks.add_mask(rule)
                logger.info("Loaded %s" % rule_id)

            except FileNotFoundError:
                logger.error("Failed to load %s" % rule_id)

        # for stage_id in stages_v2.keys():
        #     try:
        #         stage = IkaMatcher(
        #             self.mapname_left, self.mapname_top, self.mapname_width, self.mapname_height,
        #             img_file='v2_stage_%s.png' % stage_id,
        #             threshold=0.95,
        #             orig_threshold=0.30,
        #             bg_method=matcher.MM_NOT_WHITE(),
        #             fg_method=matcher.MM_WHITE(),
        #             label='stage:%s' % stage_id,
        #             call_plugins=self._call_plugins,
        #             debug=debug,
        #         )
        #         setattr(stage, 'id_', stage_id)
        #         self._stage_masks.add_mask(stage)
        #         logger.info("Loaded %s" % stage_id)

        #     except FileNotFoundError:
        #         logger.error("Failed to load %s" % stage_id)

        self._tr = TextReader()

if __name__ == "__main__":
    files = sys.argv[1:]
    for f in files:
        context = {
            'engine': {'frame': cv2.imread(f), 'msec': 1000 * 60},
            'game': {}, 'lobby': {}, 'scenes': {},
            'file': f,
        }

        obj = Spl3GameStart(None)
        obj._state_tracking(context)
