import cv2
from object_detection import ObjectDetection
import numpy as np
import math
from win import *
import numpy.typing as npt
from typing import Tuple, List

car_detector = ObjectDetection()

def detect_cars(frame: npt.NDArray, car_class:int  = 2) -> List:
    """Detect cars from frame using pre-trained model

    Args:
        frame (npt.NDArray): Frame to detect the cars from
        car_class (int, optional): car class based on model to use. Defaults to 2.

    Returns:
        List: list of cars that have been detected
    """
    class_ids, _ , boxes = car_detector.detect(frame)
    boxes = boxes[[idx for idx in range(len(class_ids)) if class_ids[idx] == car_class]]
    return boxes

def draw_box_on_cars(frame: npt.NDArray, boxes: List, center_points_cur_frame: List, color: Tuple[int, int, int] = (0, 255, 0), line_width: int = 2) -> None:
    """draw a green box on each car

    Args:
        frame (npt.NDArray): Frame to draw on
        boxes (List):  list of car boxes that have been detected
        center_points_cur_frame (List): center point to draw based on it
        color (Tuple[int, int, int], optional): color to draw the box. Defaults to (0, 255, 0).
        line_width (int, optional): width of line to draw the box. Defaults to 2.
    """
    for box in boxes:
        x, y, w, h = box
        center_x, center_y = int(x + w/2), int(y + h/2)
        center_points_cur_frame.append((center_x, center_y))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, line_width)

def detect_and_draw_cars(frame: npt.NDArray, center_points_cur_frame: List) -> None:
    """detect cars and draw boxes over them

    Args:
        frame (npt.NDArray):  Frame to draw on
        center_points_cur_frame (List):  center point to draw based on it
    """
    boxes = detect_cars(frame)
    draw_box_on_cars(frame, boxes, center_points_cur_frame)

def calc_distance(point1: Tuple[int, int], point2: Tuple[int, int])-> int:
    """Calculate the distance between two points 

    Args:
        point1 (Tuple[int, int]): first point
        point2 (Tuple[int, int]): second point

    Returns:
        int: distance between point1 and point2
    """
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

def track_cars_at_first(center_points_cur_frame: List, center_points_prev_frame: List, threshold: int = 30) -> None:
    """At second frame, i track a car if its center point is close to prev center point 

    Args:
        center_points_cur_frame (List): center point of each car at current frame
        center_points_prev_frame (List): center point of each car at prev frame
        threshold (int, optional): threshould over which i can't relate previous point to current point. Defaults to 30.
    """
    global track_id
    for point_cur in center_points_cur_frame:
        for point_prev in center_points_prev_frame:
            distance = calc_distance(point_cur, point_prev)
            if distance <= threshold:
                tracked_cars[track_id] = point_cur
                track_id +=1


def track_cars_at_rest(center_points_cur_frame: List, threshold: int = 30) -> None:
    """update center points of the cars 

    Args:
        center_points_cur_frame (List): center point of each car at current frame
        threshold (int, optional): threshould over which i can't relate previous point to current point. Defaults to 30.
    """
    global track_id
    tracked_cars_cpy = tracked_cars.copy()
    center_points_cur_frame_cpy = center_points_cur_frame.copy()
    for car_id, tracked_center_point in tracked_cars_cpy.items():
        car_exist = False
        for point_cur in center_points_cur_frame_cpy:
            distance = calc_distance(tracked_center_point, point_cur)
            if distance <= threshold:
                tracked_cars[car_id] = point_cur
                car_exist = True
                if point_cur in center_points_cur_frame:
                    center_points_cur_frame.remove(point_cur)
        if not car_exist:
            tracked_cars.pop(car_id)

    for point in center_points_cur_frame:
        tracked_cars[track_id] = point
        track_id +=1




def draw_tracked_point() -> None:
    """Draw circle and number on each car
    """
    for car_id, center_point in tracked_cars.items():
        cv2.circle(frame, center_point, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(car_id), (center_point[0], center_point[1] - 7), 0, 1, (0, 0, 255), 2)

tracked_cars = {}
center_points_prev_frame = []
frame_number = 0
track_id = 0
for frame in read_video("trial3.mp4"):
    frame_number += 1
    center_points_cur_frame = []
    detect_and_draw_cars(frame, center_points_cur_frame)
    if frame_number <= 2:
        track_cars_at_first(center_points_cur_frame, center_points_prev_frame)
    else:
        track_cars_at_rest(center_points_cur_frame)
    draw_tracked_point()
    cv2.putText(frame, f"We have tracked {track_id} unique cars till now", (0+50, 0+50), 0, 1, (0, 0, 255), 2)
    center_points_prev_frame = center_points_cur_frame.copy()
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    # break
