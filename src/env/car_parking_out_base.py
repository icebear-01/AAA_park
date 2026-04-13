import numpy as np
import pygame

from env.car_parking_base import CarParking
from env.task_utils import calc_iou, clone_map, swap_start_dest
from env.vehicle import Status
from model.agent.build_utils import normalize_unpark_img_mode
from configs import (
    UNPARK_IMG_MODE,
    UNPARK_SLOT_BUFFER,
    UNPARK_SUCCESS_IOU,
    UNPARK_IOU_STAGE_THRESHOLDS,
    UNPARK_IOU_STAGE_REWARDS,
    UNPARK_IOU_PROGRESS_SCALE,
    UNPARK_USE_SLOT_CHANNEL,
    WIN_W,
    WIN_H,
)


class CarParkingOut(CarParking):
    def __init__(self, *args, slot_buffer: float = UNPARK_SLOT_BUFFER,
                 success_iou_threshold: float = UNPARK_SUCCESS_IOU,
                 iou_stage_thresholds=UNPARK_IOU_STAGE_THRESHOLDS,
                 iou_stage_rewards=UNPARK_IOU_STAGE_REWARDS,
                 iou_progress_scale: float = UNPARK_IOU_PROGRESS_SCALE,
                 img_mode: str = UNPARK_IMG_MODE,
                 use_slot_channel: bool = UNPARK_USE_SLOT_CHANNEL,
                 **kwargs):
        self.img_mode = normalize_unpark_img_mode(img_mode, use_slot_channel=use_slot_channel)
        self.use_slot_channel = self.img_mode == "rgb_slot"
        kwargs.setdefault("enable_rs_assist", False)
        kwargs["img_extra_channels"] = int(self.use_slot_channel)
        super().__init__(*args, **kwargs)
        self.slot_buffer = slot_buffer
        self.success_iou_threshold = success_iou_threshold
        self.iou_stage_thresholds = list(iou_stage_thresholds)
        self.iou_stage_rewards = list(iou_stage_rewards)
        self.iou_progress_scale = iou_progress_scale
        if len(self.iou_stage_thresholds) != len(self.iou_stage_rewards):
            raise ValueError("Unparking IoU thresholds and rewards must have the same length")
        self.slot_box = None
        self.best_slot_iou = None
        self.triggered_iou_stages = set()

    def _reset_bookkeeping(self):
        self.reward = 0.0
        self.prev_reward = 0.0
        self.accum_arrive_reward = 0.0
        self.t = 0.0
        self.best_slot_iou = None
        self.triggered_iou_stages = set()

    def _prepare_unparking_task(self):
        # The original parking goal is the narrow slot to escape from.
        self.slot_box = self.map.dest_box
        swap_start_dest(self.map)

    def reset(self, case_id: int = None, data_dir: str = None, level: str = None):
        self._reset_bookkeeping()

        if level is not None:
            self.set_level(level)
        self.map.reset(case_id, data_dir)
        self._prepare_unparking_task()
        self.vehicle.reset(self.map.start)
        self.best_slot_iou = calc_iou(self.vehicle.box, self.slot_box)
        self.matrix = self.coord_transform_matrix()
        return self.step()[0]

    def reset_from_map(self, map_obj):
        self._reset_bookkeeping()
        self.level = getattr(map_obj, "map_level", self.level)
        self.map = clone_map(map_obj)
        self._prepare_unparking_task()
        self.vehicle.reset(self.map.start)
        self.best_slot_iou = calc_iou(self.vehicle.box, self.slot_box)
        self.matrix = self.coord_transform_matrix()
        return self.step()[0]

    def _check_arrived(self):
        if self.slot_box is None or self.t <= 1:
            return False
        return calc_iou(self.vehicle.box, self.slot_box) < self.success_iou_threshold

    def _calc_unpark_reward_terms(self):
        current_iou = calc_iou(self.vehicle.box, self.slot_box)
        prev_best_iou = current_iou if self.best_slot_iou is None else self.best_slot_iou
        best_iou_improvement = max(prev_best_iou - current_iou, 0.0)

        milestone_reward = 0.0
        triggered_milestones = []
        for threshold, reward in zip(self.iou_stage_thresholds, self.iou_stage_rewards):
            if threshold in self.triggered_iou_stages:
                continue
            if current_iou < threshold <= prev_best_iou:
                self.triggered_iou_stages.add(threshold)
                milestone_reward += reward
                triggered_milestones.append(threshold)

        self.best_slot_iou = min(prev_best_iou, current_iou)
        return current_iou, best_iou_improvement, milestone_reward, triggered_milestones, prev_best_iou

    def _mask_from_surface(self, surface):
        crop = self._capture_ego_aligned_surface(surface, background_color=(0, 0, 0))
        mask = self._surface_to_rgb_array(crop)[..., 0]
        return self.img_processor.process_mask(mask)

    def _shape_mask(self, shape_or_area):
        shape = shape_or_area.shape if hasattr(shape_or_area, "shape") else shape_or_area
        surface = pygame.Surface((WIN_W, WIN_H))
        surface.fill((0, 0, 0))
        pygame.draw.polygon(surface, (255, 255, 255), self._coord_transform(shape))
        return self._mask_from_surface(surface)

    def _obstacle_mask(self):
        surface = pygame.Surface((WIN_W, WIN_H))
        surface.fill((0, 0, 0))
        for obstacle in self.map.obstacles:
            shape = obstacle.shape if hasattr(obstacle, "shape") else obstacle
            pygame.draw.polygon(surface, (255, 255, 255), self._coord_transform(shape))
        return self._mask_from_surface(surface)

    def _occ_grid_observation(self):
        obstacle_mask = self._obstacle_mask()
        slot_mask = self._shape_mask(self.slot_box)
        ego_mask = self._shape_mask(self.vehicle.box)
        return np.concatenate((obstacle_mask, slot_mask, ego_mask), axis=-1)

    def _process_img_observation(self, img):
        if self.img_mode == "occ_grid":
            return self._occ_grid_observation()
        return super()._process_img_observation(img)

    def _get_img_extra_channels(self):
        if not (self.use_img_observation and self.use_slot_channel and self.slot_box is not None):
            return None

        return self._shape_mask(self.slot_box)

    def step(self, action=None):
        observation, reward_info, status, info = super().step(action)

        current_iou, best_iou_improvement, milestone_reward, triggered_milestones, prev_best_iou = \
            self._calc_unpark_reward_terms()

        reward_info['rs_dist_reward'] = 0.0
        reward_info['dist_reward'] = best_iou_improvement * self.iou_progress_scale
        reward_info['angle_reward'] = 0.0
        reward_info['box_union_reward'] = milestone_reward

        info['slot_iou'] = current_iou
        info['best_slot_iou'] = self.best_slot_iou
        info['prev_best_slot_iou'] = prev_best_iou
        info['iou_progress_reward'] = reward_info['dist_reward']
        info['iou_milestone_reward'] = milestone_reward
        info['iou_milestones'] = triggered_milestones
        info['reward_bonus'] = 0.0

        # The generic wrapper ignores reward_info on terminal steps, so preserve
        # one-shot milestone rewards if they are first reached on the success step.
        if status == Status.ARRIVED and milestone_reward > 0:
            info['reward_bonus'] = milestone_reward

        return observation, reward_info, status, info
