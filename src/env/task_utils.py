from copy import deepcopy

from shapely.geometry import Polygon


def swap_start_dest(map_obj):
    map_obj.start, map_obj.dest = map_obj.dest, map_obj.start
    map_obj.start_box, map_obj.dest_box = map_obj.dest_box, map_obj.start_box
    return map_obj


def clone_map(map_obj):
    return deepcopy(map_obj)


def build_slot_exit_zone(slot_box, clearance: float):
    return Polygon(slot_box).buffer(clearance, cap_style=2, join_style=2)


def calc_iou(shape_a, shape_b):
    polygon_a = Polygon(shape_a)
    polygon_b = Polygon(shape_b)
    union_area = polygon_a.union(polygon_b).area
    if union_area <= 1e-8:
        return 0.0
    return polygon_a.intersection(polygon_b).area / union_area
