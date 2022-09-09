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
import copy
import blend_modes

inklings_y_offset = 10

inklings_team1_x_offset = 328
inklings_team2_x_offset = 695

inklings_box_width = 260
inklings_box_height = 80

bar_y_offset = 50
bar_height = 10

bar_check_extent_width = inklings_box_width

team_advantage_size = (241, 78)
team_neutral_size = (236, 76)
team_disadvantage_size = (220, 71)
team_danger_size = (207, 70)

splatted_y_trim = 20

tentacle_width_neutral = 41

def _kmeans_cluster(data, initial_cluster_count, distance):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    cluster_count = min(len(data), initial_cluster_count)
    check_clusters = cluster_count > 0
    clusters = data
    x_coordinates = []
    while check_clusters and cluster_count > 0:
        ret,label,clusters = cv2.kmeans(np.float32(data), cluster_count, None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
        x_coordinates = np.sort(clusters[:, 0])
        if len(x_coordinates) == 1 or x_coordinates[-1] - x_coordinates[-2] > distance:
            check_clusters = False
        cluster_count = cluster_count - 1
    return clusters, x_coordinates


"""
Entry extraction
"""
def get_tentacle_width(img_team, team_index):
    if team_index == 0:
        img_team = cv2.flip(img_team, 1)
    else:
        img_team = copy.deepcopy(img_team)
    tentacle_width = None

    x_tentacles = find_vertical_edges(img_team, team_index)
    x_splatted = find_splatted(img_team, team_index)

    # remove tentacle lines that appear under a splatted X
    for splat_center in x_splatted:
       x_tentacles = [x for x in x_tentacles if x < splat_center - tentacle_width_neutral/3 or x > splat_center + tentacle_width_neutral/3] 

    if len(x_tentacles) >= 2:
        diffs = np.diff(x_tentacles)
        diffs = [d for d in diffs if d > tentacle_width_neutral - 7 and d < tentacle_width_neutral + 7]
        if len(diffs) > 0 :
            clusters, x_coordinates = _kmeans_cluster(diffs, 4, 6)
            tentacle_width = clusters[0][0]
            return tentacle_width

    # If we weren't able to work out the distance from the tentacles, check the splat locations    
    if len(x_splatted) > 1:
        diffs = np.diff(x_splatted)
        diffs = [d for d in diffs if d > tentacle_width_neutral - 7 and d < tentacle_width_neutral + 7]
        if len(diffs) > 0 :
            diffs = np.bincount(diffs)
            if np.max(diffs) > 1:
                tentacle_width = np.argmax(diffs)
                return tentacle_width    

    return tentacle_width

def find_vertical_edges(img_team, team_index):
    x_offset = 0 if team_index == 0 else inklings_box_width - bar_check_extent_width
    bar = img_team[bar_y_offset:bar_y_offset+bar_height, x_offset:x_offset+bar_check_extent_width].astype(np.uint8)

    resized = cv2.resize(bar, (bar_check_extent_width, bar_height * 3), interpolation=cv2.INTER_AREA)
    resized = cv2.GaussianBlur(resized, (3,17), 0)


    bar_edges = cv2.Canny(image=resized, threshold1=0, threshold2=200) 
    bar_edges = cv2.resize(bar_edges, (bar_check_extent_width, bar_height))
    contours, _ = cv2.findContours(bar_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    vertical_alignments = []
    other_contours = []
    for contour in contours:
        y_coordinates = contour[:, 0, 1]
        x_coordinates = contour[:, 0, 0]
        x_avg = np.median(x_coordinates)
        spans_height = np.max(y_coordinates) - np.min(y_coordinates) > bar_height - 3
        is_vertical = np.max(x_coordinates) - np.min(x_coordinates) < 3

        if is_vertical:
            if spans_height:
                # contour = contour[0:2,:,:]
                # contour[0, 0] = [x_avg, 0]
                # contour[1, 0] = [x_avg, bar_height - 1]
                vertical_alignments.append(int(x_avg))
            else:
                other_contours.append(contour)
        else:
            other_contours.append(contour)
    return np.sort(vertical_alignments)

def find_splatted(img_team, team_index):
    # To avoid evaluating possible grey background noise, trim the box height
    img_team = img_team.copy()[splatted_y_trim:inklings_box_height-splatted_y_trim, :]
    # Create image with just the crosses highlighted
    mask_inklings_cross = cv2.inRange(img_team, (80, 80, 80), (120, 120, 120))
    crosses = np.zeros(img_team.shape, dtype=np.uint8)
    crosses[mask_inklings_cross > 0] = (255, 255, 255)
    crosses_grey = cv2.cvtColor(crosses, cv2.COLOR_RGB2GRAY)

    lines = cv2.HoughLines(crosses_grey, rho=1, theta=np.pi / 180, threshold=65, )
 
    linesNWSE = []
    linesSWNE = []
    intersections = []
    if lines is not None:
        for line in lines:
            theta = line[0][1]
            if int(theta * 360 / np.pi)  == 270:
                linesNWSE.append(line)
            elif int(theta * 360 / np.pi)  == 90:
                linesSWNE.append(line)

    # drawnlines = img_team.copy()
    for line in linesNWSE:
        rho1, theta1 = line[0]
        a = np.cos(theta1)
        b = np.sin(theta1)
        x0 = a * rho1
        y0 = b * rho1
        # pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        # pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        # cv2.line(drawnlines, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)
        for crossLine in linesSWNE:
            rho2, theta2 = crossLine[0]
            a = np.cos(theta2)
            b = np.sin(theta2)
            x0 = a * rho2
            y0 = b * rho2
            # pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            # pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            # cv2.line(drawnlines, pt1, pt2, (255,0,0), 3, cv2.LINE_AA)
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            if y0 >= 14 and y0 <= 18:
                intersections.append([x0, y0])

    clusters, x_coordinates = _kmeans_cluster(intersections, 4, 10)
    return x_coordinates


def transform_inklings(frame, context):
    # Given an image, determine the weighting of the squid sizes so that details can be determined
    team1 = frame[inklings_y_offset: inklings_y_offset + inklings_box_height, inklings_team1_x_offset : inklings_team1_x_offset + inklings_box_width]
    team2 = frame[inklings_y_offset: inklings_y_offset + inklings_box_height, inklings_team2_x_offset : inklings_team2_x_offset + inklings_box_width]

    team1_weight = get_tentacle_width(team1, 0)
    team2_weight = get_tentacle_width(team2, 1)
    if team1_weight is None and team2_weight is None:
        # guess
        team1_weight = 41
        team2_weight = 41
    if team1_weight is None:
        team1_weight = (41/team2_weight) * 40
    if team2_weight is None:
        team2_weight = (41/team1_weight) * 40

    # Force max/min weights
    if team1_weight > 46:
        team1_weight = 46
    if team2_weight > 46:
        team2_weight = 46

    team1_height = int(team1_weight*1.74)
    team1_width = int(4*team1_weight*1.41)
    team1_offset_x = int(40 - team1_weight * 4 / 5)
    team1_offset_y = int(36 - team1_weight * 5 / 7)

    team2_height = int(team2_weight*1.74)
    team2_width = int(4*team2_weight*1.41)
    team2_offset_x = int(27 - team2_weight * 4 / 7)
    team2_offset_y = int(36 - team2_weight * 5 / 7)

    return {
        'scenario': 'other',
        'team1': cv2.resize(team1[team1_offset_y:team1_offset_y+team1_height,team1_offset_x:team1_offset_x+team1_width], (inklings_box_width, inklings_box_height)),
        'team2': cv2.resize(team2[team2_offset_y:team2_offset_y+team2_height,team2_offset_x:team2_offset_x+team2_width], (inklings_box_width, inklings_box_height)),
    }


if __name__ == '__main__':
    import sys

    img = cv2.imread(sys.argv[1], 1)
    context = {
        'game' : {
            'team_color_rgb': [
                # (123, 3, 147), # PURPLE
                # (67, 186, 5), # LUMIGREEN
                # (41, 34, 181),
                # (94, 182, 4), # GREEN
                # (217, 193, 0), # YELLOW
                # (0, 122, 201), # LIGHTBLUE
                (68, 215, 17), # LIME GREEN
                (187, 41, 121), # MAGENTA
            ],
        }
    }
    r = transform_inklings(img, context)

    print(r['scenario'])
    cv2.imshow('left', r['team1'])
    cv2.waitKey()
    cv2.imshow('right', r['team2'])
    cv2.waitKey()
