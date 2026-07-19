import numpy as np
import os
import robosuite.utils.transform_utils as T

from copy import deepcopy
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import CustomMaterial
import robosuite.macros as macros

import mujoco

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs.robots import *
from libero.libero.envs.utils import *
from libero.libero.envs.object_states import *
from libero.libero.envs.objects import *
from libero.libero.envs.regions import *
from libero.libero.envs.arenas import *


DIR_PATH = os.path.dirname(os.path.realpath(__file__))

TASK_MAPPING = {}


def register_problem(target_class):
    """We design the mapping to be case-INsensitive."""
    TASK_MAPPING[target_class.__name__.lower()] = target_class


import time


class BDDLBaseDomain(SingleArmEnv):
    """
    A base domain for parsing bddl files.
    """

    def __init__(
        self,
        bddl_file_name,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        use_latch=False,
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        object_property_initializers=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,
        renderer="mujoco",
        table_full_size=(1.0, 1.0, 0.05),
        workspace_offset=(0.0, 0.0, 0.0),
        arena_type="table",
        scene_xml="scenes/libero_base_style.xml",
        scene_properties={},
        **kwargs,
    ):
        t0 = time.time()
        # settings for table top (hardcoded since it's not an essential part of the environment)
        self.workspace_offset = workspace_offset
        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.conditional_placement_initializer = None
        self.conditional_placement_on_objects_initializer = None

        # object property initializer

        if object_property_initializers is not None:
            self.object_property_initializers = object_property_initializers
        else:
            self.object_property_initializers = list()

        # Keep track of movable objects in the tasks
        self.objects_dict = {}
        # Kepp track of fixed objects in the tasks
        self.fixtures_dict = {}
        # Keep track of site objects in the tasks. site objects
        # (instances of SiteObject)
        self.object_sites_dict = {}
        # This is a dictionary that stores all the object states
        # interface for all the objects
        self.object_states_dict = {}

        # For those that require visual feature changes, update the state every time step to avoid missing state changes. We keep track of this type of objects to make predicate checking more efficient.
        self.tracking_object_states_change = []

        self.object_sites_dict = {}

        self.objects = []
        self.fixtures = []
        # self.custom_material_dict = {}

        self.custom_asset_dir = os.path.abspath(os.path.join(DIR_PATH, "../assets"))

        self.bddl_file_name = bddl_file_name
        self.parsed_problem = BDDLUtils.robosuite_parse_problem(self.bddl_file_name)

        self.obj_of_interest = self.parsed_problem["obj_of_interest"]

        self._assert_problem_name()

        self._arena_type = arena_type
        self._arena_xml = os.path.join(self.custom_asset_dir, scene_xml)
        self._arena_properties = scene_properties

        # 需在 super().__init__ 前初始化：父类初始化流程会触发 _reset_internal()
        # 而 _reset_internal() 会调用 suction 相关状态函数。
        self._init_suction_state()

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            **kwargs,
        )

    def seed(self, seed):
        np.random.seed(seed)

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 1.0 is provided if the task succeeds.

        Args:
            action (np.array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 1.0

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 1.0

        return reward

    def _assert_problem_name(self):
        """Implement this to make sure the loaded bddl file has the correct problem name specification."""
        assert (
            self.parsed_problem["problem_name"] == self.__class__.__name__.lower()
        ), "Problem name mismatched"

    def _load_fixtures_in_arena(self, mujoco_arena):
        """
        Load fixtures based on the bddl file description. Please override the method in the custom problem file.
        """
        raise NotImplementedError

    def _load_objects_in_arena(self, mujoco_arena):
        """
        Load movable objects based on the bddl file description
        """
        raise NotImplementedError

    def _load_sites_in_arena(self, mujoco_arena):
        """
        Load sites information from each object to keep track of them for predicate checking
        """
        raise NotImplementedError

    def _generate_object_state_wrapper(
        self, skip_object_names=["main_table", "floor", "countertop", "coffee_table"]
    ):
        object_states_dict = {}
        tracking_object_states_changes = []
        for object_name in self.objects_dict.keys():
            if object_name in skip_object_names:
                continue
            object_states_dict[object_name] = ObjectState(self, object_name)
            if (
                self.objects_dict[object_name].category_name
                in VISUAL_CHANGE_OBJECTS_DICT
            ):
                tracking_object_states_changes.append(object_states_dict[object_name])

        for object_name in self.fixtures_dict.keys():
            if object_name in skip_object_names:
                continue
            object_states_dict[object_name] = ObjectState(
                self, object_name, is_fixture=True
            )
            if (
                self.fixtures_dict[object_name].category_name
                in VISUAL_CHANGE_OBJECTS_DICT
            ):
                tracking_object_states_changes.append(object_states_dict[object_name])

        for object_name in self.object_sites_dict.keys():
            if object_name in skip_object_names:
                continue
            object_states_dict[object_name] = SiteObjectState(
                self,
                object_name,
                parent_name=self.object_sites_dict[object_name].parent_name,
            )
        self.object_states_dict = object_states_dict
        self.tracking_object_states_change = tracking_object_states_changes

    def _load_distracting_objects(self, mujoco_arena):
        raise NotImplementedError

    def _load_custom_material(self):
        """
        Define all the textures
        """
        # self.custom_material_dict = dict()

        # tex_attrib = {
        #     "type": "cube"
        # }

        # self.custom_material_dict["bread"] = CustomMaterial(
        #     texture="Bread",
        #     tex_name="bread",
        #     mat_name="MatBread",
        #     tex_attrib=tex_attrib,
        #     mat_attrib={"texrepeat": "3 3", "specular": "0.4","shininess": "0.1"}
        # )

    def _setup_camera(self, mujoco_arena):
        # Modify default agentview camera
        mujoco_arena.set_camera(
            camera_name="canonical_agentview",
            pos=[0.5386131746834771, 0.0, 1.4903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.5886131746834771, 0.0, 1.4903500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()
        # Adjust base pose accordingly

        if self._arena_type == "table":
            xpos = self.robots[0].robot_model.base_xpos_offset["table"](
                self.table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = TableArena(
                table_full_size=self.table_full_size,
                table_offset=self.workspace_offset,
                table_friction=(0.6, 0.005, 0.0001),
                xml=self._arena_xml,
                **self._arena_properties,
            )
        elif self._arena_type == "kitchen":
            xpos = self.robots[0].robot_model.base_xpos_offset["kitchen_table"](
                self.kitchen_table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = KitchenTableArena(
                table_full_size=self.kitchen_table_full_size,
                table_offset=self.workspace_offset,
                xml=self._arena_xml,
                **self._arena_properties,
            )

        elif self._arena_type == "floor":
            xpos = self.robots[0].robot_model.base_xpos_offset["empty"]
            self.robots[0].robot_model.set_base_xpos(xpos)

            mujoco_arena = EmptyArena(
                xml=self._arena_xml,
                **self._arena_properties,
            )
        elif self._arena_type == "coffee_table":
            xpos = self.robots[0].robot_model.base_xpos_offset["coffee_table"](
                self.coffee_table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = CoffeeTableArena(
                xml=self._arena_xml,
                **self._arena_properties,
            )

        elif self._arena_type == "living_room":
            xpos = self.robots[0].robot_model.base_xpos_offset["living_room_table"](
                self.living_room_table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = LivingRoomTableArena(
                xml=self._arena_xml,
                **self._arena_properties,
            )

        elif self._arena_type == "study":
            xpos = self.robots[0].robot_model.base_xpos_offset["study_table"](
                self.study_table_full_size[0]
            )
            self.robots[0].robot_model.set_base_xpos(xpos)
            mujoco_arena = StudyTableArena(
                xml=self._arena_xml,
                **self._arena_properties,
            )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self._setup_camera(mujoco_arena)

        self._load_custom_material()

        self._load_fixtures_in_arena(mujoco_arena)

        self._load_objects_in_arena(mujoco_arena)

        self._load_sites_in_arena(mujoco_arena)

        self._generate_object_state_wrapper()

        self._setup_placement_initializer(mujoco_arena)

        self.objects = list(self.objects_dict.values())
        self.fixtures = list(self.fixtures_dict.values())

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.objects + self.fixtures,
        )

        for fixture in self.fixtures:
            self.model.merge_assets(fixture)

    def _setup_placement_initializer(self, mujoco_arena):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.conditional_placement_initializer = SiteSequentialCompositeSampler(
            name="ConditionalSiteSampler"
        )
        self.conditional_placement_on_objects_initializer = SequentialCompositeSampler(
            name="ConditionalObjectSampler"
        )
        self._add_placement_initializer()

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.obj_body_id = dict()

        for (object_name, object_body) in self.objects_dict.items():
            self.obj_body_id[object_name] = self.sim.model.body_name2id(
                object_body.root_body
            )

        for (fixture_name, fixture_body) in self.fixtures_dict.items():
            self.obj_body_id[fixture_name] = self.sim.model.body_name2id(
                fixture_body.root_body
            )

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        observables["robot0_joint_pos"]._active = True

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            sensors = []
            names = [s.__name__ for s in sensors]

            # Also append handle qpos if we're using a locked drawer version with rotatable handle

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality="object")
        def world_pose_in_gripper(obs_cache):
            return (
                T.pose_inv(
                    T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))
                )
                if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache
                else np.eye(4)
            )

        sensors.append(world_pose_in_gripper)
        names.append("world_pose_in_gripper")

        for (i, obj) in enumerate(self.objects):
            obj_sensors, obj_sensor_names = self._create_obj_sensors(
                obj_name=obj.name, modality="object"
            )

            sensors += obj_sensors
            names += obj_sensor_names

        for name, s in zip(names, sensors):
            if name == "world_pose_in_gripper":
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                    enabled=True,
                    active=False,
                )
            else:
                observables[name] = Observable(
                    name=name, sensor=s, sampling_rate=self.control_freq
                )

        # ========== 添加吸盘六维力传感器 ==========
        # 这里对外只暴露一个主观测键名：suction_wrench。
        # 当前主语义是统一的“吸盘端准静态外载荷”：
        # - 排除吸盘自身工具重量；
        # - 排除速度 / 加速度造成的动态惯性扰动；
        # - 保留外界接触力、被承载物体重力等真实准静态外载荷。
        #
        # 注意：吸盘自身工具重量仍然不会进入主显示值。
        # raw / gravity-compensated / contact 等中间量会继续保留在 info 里做调试。
        @sensor(modality=f"{pf}proprio")
        def suction_wrench(obs_cache):
            wrench = self._get_suction_wrench()
            if wrench is None:
                return np.zeros(6, dtype=np.float64)
            return wrench

        # 注册统一后的吸盘六维力观测。
        observables["suction_wrench"] = Observable(
            name="suction_wrench",
            sensor=suction_wrench,
            sampling_rate=self.control_freq,
        )
        # ========== 六维力传感器添加完成 ==========

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """
        Helper function to create sensors for a given object. This is abstracted in a separate function call so that we
        don't have local function naming collisions during the _setup_observables() call.

        Args:
            obj_name (str): Name of object to create sensors for
            modality (str): Modality to assign to all sensors

        Returns:
            2-tuple:
                sensors (list): Array of sensors for the given obj
                names (list): array of corresponding observable names
        """
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(
                self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw"
            )

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            # Immediately return default value if cache is empty
            if any(
                [
                    name not in obs_cache
                    for name in [
                        f"{obj_name}_pos",
                        f"{obj_name}_quat",
                        "world_pose_in_gripper",
                    ]
                ]
            ):
                return np.zeros(3)
            obj_pose = T.pose2mat(
                (obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"])
            )
            rel_pose = T.pose_in_A_to_pose_in_B(
                obj_pose, obs_cache["world_pose_in_gripper"]
            )
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return (
                obs_cache[f"{obj_name}_to_{pf}eef_quat"]
                if f"{obj_name}_to_{pf}eef_quat" in obs_cache
                else np.zeros(4)
            )

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [
            f"{obj_name}_pos",
            f"{obj_name}_quat",
            f"{obj_name}_to_{pf}eef_pos",
            f"{obj_name}_to_{pf}eef_quat",
        ]

        return sensors, names

    def _add_placement_initializer(self):

        mapping_inv = {}
        for k, values in self.parsed_problem["fixtures"].items():
            for v in values:
                mapping_inv[v] = k
        for k, values in self.parsed_problem["objects"].items():
            for v in values:
                mapping_inv[v] = k

        regions = self.parsed_problem["regions"]
        initial_state = self.parsed_problem["initial_state"]
        problem_name = self.parsed_problem["problem_name"]

        conditioned_initial_place_state_on_sites = []
        conditioned_initial_place_state_on_objects = []
        conditioned_initial_place_state_in_objects = []

        for state in initial_state:
            if state[0] == "on" and state[2] in self.objects_dict:
                conditioned_initial_place_state_on_objects.append(state)
                continue

            # (Yifeng) Given that an object needs to have a certain "containing" region in order to hold the relation "In", we assume that users need to specify the containing region of the object already.
            if state[0] == "in" and state[2] in regions:
                conditioned_initial_place_state_in_objects.append(state)
                continue
            # Check if the predicate is in the form of On(object, region)
            if state[0] == "on" and state[2] in regions:
                object_name = state[1]
                region_name = state[2]
                target_name = regions[region_name]["target"]
                x_ranges, y_ranges = rectangle2xyrange(regions[region_name]["ranges"])
                yaw_rotation = regions[region_name]["yaw_rotation"]
                if (
                    target_name in self.objects_dict
                    or target_name in self.fixtures_dict
                ):
                    conditioned_initial_place_state_on_sites.append(state)
                    continue
                if self.is_fixture(object_name):
                    # This is to place environment fixtures.
                    fixture_sampler = MultiRegionRandomSampler(
                        f"{object_name}_sampler",
                        mujoco_objects=self.fixtures_dict[object_name],
                        x_ranges=x_ranges,
                        y_ranges=y_ranges,
                        rotation=yaw_rotation,
                        rotation_axis="z",
                        z_offset=self.z_offset,  # -self.table_full_size[2],
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False,
                        reference_pos=self.workspace_offset,
                    )
                    self.placement_initializer.append_sampler(fixture_sampler)
                else:
                    # This is to place movable objects.
                    region_sampler = get_region_samplers(
                        problem_name, mapping_inv[target_name]
                    )(
                        object_name,
                        self.objects_dict[object_name],
                        x_ranges=x_ranges,
                        y_ranges=y_ranges,
                        rotation=self.objects_dict[object_name].rotation,
                        rotation_axis=self.objects_dict[object_name].rotation_axis,
                        reference_pos=self.workspace_offset,
                    )
                    self.placement_initializer.append_sampler(region_sampler)
            if state[0] in ["open", "close"]:
                # If "open" is implemented, we assume "close" is also implemented
                if state[1] in self.object_states_dict and hasattr(
                    self.object_states_dict[state[1]], "set_joint"
                ):
                    obj = self.get_object(state[1])
                    if state[0] == "open":
                        joint_ranges = obj.object_properties["articulation"][
                            "default_open_ranges"
                        ]
                    else:
                        joint_ranges = obj.object_properties["articulation"][
                            "default_close_ranges"
                        ]

                    property_initializer = OpenCloseSampler(
                        name=obj.name,
                        state_type=state[0],
                        joint_ranges=joint_ranges,
                    )
                    self.object_property_initializers.append(property_initializer)
            elif state[0] in ["turnon", "turnoff"]:
                # If "turnon" is implemented, we assume "turnoff" is also implemented.
                if state[1] in self.object_states_dict and hasattr(
                    self.object_states_dict[state[1]], "set_joint"
                ):
                    obj = self.get_object(state[1])
                    if state[0] == "turnon":
                        joint_ranges = obj.object_properties["articulation"][
                            "default_turnon_ranges"
                        ]
                    else:
                        joint_ranges = obj.object_properties["articulation"][
                            "default_turnoff_ranges"
                        ]

                    property_initializer = TurnOnOffSampler(
                        name=obj.name,
                        state_type=state[0],
                        joint_ranges=joint_ranges,
                    )
                    self.object_property_initializers.append(property_initializer)

        # Place objects that are on sites
        for state in conditioned_initial_place_state_on_sites:
            object_name = state[1]
            region_name = state[2]
            target_name = regions[region_name]["target"]
            site_xy_size = self.object_sites_dict[region_name].size[:2]
            sampler = SiteRegionRandomSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                x_ranges=[[-site_xy_size[0] / 2, site_xy_size[0] / 2]],
                y_ranges=[[-site_xy_size[1] / 2, site_xy_size[1] / 2]],
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_initializer.append_sampler(
                sampler, {"reference": target_name, "site_name": region_name}
            )
        # Place objects that are on other objects
        for state in conditioned_initial_place_state_on_objects:
            object_name = state[1]
            other_object_name = state[2]
            sampler = ObjectBasedSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                x_ranges=[[0.0, 0.0]],
                y_ranges=[[0.0, 0.0]],
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=False,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_on_objects_initializer.append_sampler(
                sampler, {"reference": other_object_name}
            )
        # Place objects inside some containing regions
        for state in conditioned_initial_place_state_in_objects:
            object_name = state[1]
            region_name = state[2]
            target_name = regions[region_name]["target"]

            site_xy_size = self.object_sites_dict[region_name].size[:2]
            sampler = InSiteRegionRandomSampler(
                f"{object_name}_sampler",
                mujoco_objects=self.objects_dict[object_name],
                # x_ranges=[[-site_xy_size[0] / 2, site_xy_size[0] / 2]],
                # y_ranges=[[-site_xy_size[1] / 2, site_xy_size[1] / 2]],
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                rotation=self.objects_dict[object_name].rotation,
                rotation_axis=self.objects_dict[object_name].rotation_axis,
            )
            self.conditional_placement_initializer.append_sampler(
                sampler, {"reference": target_name, "site_name": region_name}
            )

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self._release_suction_latch()
        self._suction_enabled = False
        self._suction_active = False
        self._suction_geom_id = None
        self._suction_body_id = None
        self._suction_site_id = None
        self._suction_eq_id = None
        self._suction_force_sensor_id = None
        self._suction_torque_sensor_id = None
        self._force_body_ids = None
        self._suction_candidate_body_id = None
        self._suction_candidate_joint_id = None
        self._suction_contact_stable_steps = 0
        self._suction_display_wrench_world = np.zeros(6, dtype=np.float64)
        self._suction_contact_wrench_world = np.zeros(6, dtype=np.float64)
        self._suction_external_contact_wrench_world = np.zeros(6, dtype=np.float64)
        self._suction_payload_gravity_wrench_world = np.zeros(6, dtype=np.float64)
        self._suction_payload_quasistatic_wrench_world = np.zeros(6, dtype=np.float64)
        self._suction_latched_gravity_wrench_world = np.zeros(6, dtype=np.float64)
        self._suction_debug = {
            "ncon": 0,
            "cup_contact": False,
            "latched": False,
            "latched_joint_type": None,
            "candidate_body_id": -1,
            "stable_steps": 0,
            "relative_speed": None,
            "overlap_ratio": -1.0,
            "overlap_passed": False,
            "overlap_contact_geom_id": -1,
        }

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            for object_property_initializer in self.object_property_initializers:
                if isinstance(object_property_initializer, OpenCloseSampler):
                    joint_pos = object_property_initializer.sample()
                    self.object_states_dict[object_property_initializer.name].set_joint(
                        joint_pos
                    )
                elif isinstance(object_property_initializer, TurnOnOffSampler):
                    joint_pos = object_property_initializer.sample()
                    self.object_states_dict[object_property_initializer.name].set_joint(
                        joint_pos
                    )
                else:
                    print("Warning!!! This sampler doesn't seem to be used")
            # robosuite didn't provide api for this stepping. we manually do this stepping to increase the speed of resetting simulation.
            mujoco.mj_step1(self.sim.model._model, self.sim.data._data)

            object_placements = self.placement_initializer.sample()
            object_placements = self.conditional_placement_initializer.sample(
                self.sim, object_placements
            )
            object_placements = (
                self.conditional_placement_on_objects_initializer.sample(
                    object_placements
                )
            )
            for obj_pos, obj_quat, obj in object_placements.values():
                if obj.name not in list(self.fixtures_dict.keys()):
                    # This is for movable object resetting
                    self.sim.data.set_joint_qpos(
                        obj.joints[-1],
                        np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                    )
                else:
                    # This is for fixture resetting
                    body_id = self.sim.model.body_name2id(obj.root_body)
                    self.sim.model.body_pos[body_id] = obj_pos
                    self.sim.model.body_quat[body_id] = obj_quat

    def _check_success(self):
        """
        This needs to match with the goal description from the bddl file

        Returns:
            bool: True if drawer has been opened
        """
        return False

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the drawer handle.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def step(self, action):
        if self.action_dim == 4 and len(action) > 4:
            # Convert OSC_POSITION action
            action = np.array(action)
            action = np.concatenate((action[:3], action[-1:]), axis=-1)

        obs, reward, done, info = super().step(action)
        done = self._check_success()

        return obs, reward, done, info

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)

    def _post_action(self, action):
        reward, done, info = super()._post_action(action)

        self._update_suction(action)

        self._update_suction_wrench_cache(update_bias=True)
        suction_wrench = np.asarray(self._suction_display_wrench_world, dtype=np.float64).copy()

        info["suction_on"] = bool(self._suction_enabled)
        info["suction_enabled"] = bool(self._suction_enabled)
        info["suction_latched"] = bool(self._suction_latched)
        info["suction_latched_body_id"] = int(self._latched_body_id) if self._latched_body_id is not None else -1
        info["suction_eq_id"] = int(self._suction_eq_id) if self._suction_eq_id is not None else -1
        info["suction_ncon"] = int(self._suction_debug.get("ncon", 0))
        info["suction_cup_contact"] = bool(self._suction_debug.get("cup_contact", False))
        info["suction_latched_joint_type"] = self._suction_debug.get("latched_joint_type", None)
        info["suction_candidate_body_id"] = int(self._suction_debug.get("candidate_body_id", -1))
        info["suction_stable_steps"] = int(self._suction_debug.get("stable_steps", 0))
        info["suction_relative_speed"] = self._suction_debug.get("relative_speed", None)
        info["suction_overlap_ratio"] = float(self._suction_debug.get("overlap_ratio", -1.0))
        info["suction_overlap_passed"] = bool(self._suction_debug.get("overlap_passed", False))
        info["suction_overlap_contact_geom_id"] = int(self._suction_debug.get("overlap_contact_geom_id", -1))
        info["suction_force_torque"] = suction_wrench.copy()
        info["suction_wrench_display_world"] = np.asarray(self._suction_display_wrench_world, dtype=np.float64).copy()
        info["suction_wrench_external_contact_world"] = np.asarray(
            self._suction_external_contact_wrench_world, dtype=np.float64
        ).copy()
        info["suction_wrench_payload_gravity_world"] = np.asarray(
            self._suction_payload_gravity_wrench_world, dtype=np.float64
        ).copy()
        info["suction_wrench_payload_quasistatic_world"] = np.asarray(
            self._suction_payload_quasistatic_wrench_world, dtype=np.float64
        ).copy()
        info["suction_wrench_quasistatic_world"] = np.asarray(
            self._suction_payload_quasistatic_wrench_world, dtype=np.float64
        ).copy()
        info["suction_wrench_contact_world"] = np.asarray(self._suction_contact_wrench_world, dtype=np.float64).copy()
        info["suction_wrench_latched_gravity_world"] = np.asarray(
            self._suction_latched_gravity_wrench_world, dtype=np.float64
        ).copy()
        info["suction_wrench_local_raw"] = np.asarray(self._suction_wrench_local_raw, dtype=np.float64).copy()
        info["suction_wrench_world_raw"] = np.asarray(self._suction_wrench_world_raw, dtype=np.float64).copy()
        info["suction_wrench_gravity_world"] = np.asarray(self._suction_wrench_gravity_world, dtype=np.float64).copy()
        info["suction_wrench_bias"] = np.asarray(self._suction_bias_world, dtype=np.float64).copy()
        info["suction_wrench_total_load_world"] = np.asarray(self._suction_wrench_compensated, dtype=np.float64).copy()
        info["suction_wrench_gravity_sign"] = (
            float(self._suction_gravity_sign) if self._suction_gravity_sign is not None else 0.0
        )

        self._post_process()

        return reward, done, info

    def _post_process(self):
        # Update some object states, such as light switching etc.
        for object_state in self.tracking_object_states_change:
            object_state.update_state()

    def _init_suction_state(self):
        self._suction_enabled = False
        self._suction_active = False
        self._suction_latched = False
        self._latched_body_id = None
        self._latched_joint_id = None
        self._suction_eq_id = None
        self._suction_candidate_body_id = None
        self._suction_candidate_joint_id = None
        self._suction_contact_stable_steps = 0
        self._suction_required_stable_steps = 1
        self._suction_required_overlap_ratio = 0.9
        self._suction_overlap_projection_depth = 0.02
        self._suction_overlap_ray_start_offset = 0.001
        self._suction_overlap_sample_count = 64
        self._suction_overlap_sample_offsets = None

        self._suction_geom_id = None
        self._suction_body_id = None
        self._suction_site_id = None
        self._suction_force_sensor_id = None
        self._suction_torque_sensor_id = None
        self._force_body_ids = None
        self._suction_tool_body_ids = None
        self._suction_tool_mass = None
        self._suction_gravity_sign = None
        self._suction_bias_world = np.zeros(6, dtype=np.float64)
        self._suction_bias_initialized = False
        self._suction_bias_alpha = 0.02
        self._suction_display_wrench_world = np.zeros(6, dtype=np.float64)
        self._suction_contact_wrench_world = np.zeros(6, dtype=np.float64)
        self._suction_external_contact_wrench_world = np.zeros(6, dtype=np.float64)
        self._suction_payload_gravity_wrench_world = np.zeros(6, dtype=np.float64)
        self._suction_payload_quasistatic_wrench_world = np.zeros(6, dtype=np.float64)
        self._suction_latched_gravity_wrench_world = np.zeros(6, dtype=np.float64)
        self._suction_wrench_local_raw = np.zeros(6, dtype=np.float64)
        self._suction_wrench_world_raw = np.zeros(6, dtype=np.float64)
        self._suction_wrench_gravity_world = np.zeros(6, dtype=np.float64)
        self._suction_wrench_compensated = np.zeros(6, dtype=np.float64)
        self._suction_debug = {
            "ncon": 0,
            "cup_contact": False,
            "latched": False,
            "latched_joint_type": None,
            "candidate_body_id": -1,
            "stable_steps": 0,
            "relative_speed": None,
            "overlap_ratio": -1.0,
            "overlap_passed": False,
            "overlap_contact_geom_id": -1,
        }

    def _safe_body_name2id(self, name):
        try:
            return int(self.sim.model.body_name2id(name))
        except ValueError:
            return None

    def _safe_geom_name2id(self, name):
        try:
            return int(self.sim.model.geom_name2id(name))
        except ValueError:
            return None

    def _safe_site_name2id(self, name):
        try:
            return int(self.sim.model.site_name2id(name))
        except ValueError:
            return None

    def _find_geom_id_by_substring(self, token):
        raw_model = self.sim.model._model
        for geom_id in range(int(raw_model.ngeom)):
            geom_name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name is not None and token in geom_name:
                return geom_id
        return None

    def _safe_sensor_name2id(self, name):
        try:
            return int(self.sim.model.sensor_name2id(name))
        except ValueError:
            return None

    def _find_sensor_id_by_substring(self, token):
        raw_model = self.sim.model._model
        for sensor_id in range(int(raw_model.nsensor)):
            sensor_name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_id)
            if sensor_name is not None and token in sensor_name:
                return sensor_id
        return None

    def _resolve_suction_handles(self):
        if self._suction_geom_id is not None and self._suction_body_id is not None:
            return

        geom_id = self._safe_geom_name2id("gripper0_suction_cup_collision")
        if geom_id is None:
            geom_id = self._safe_geom_name2id("gripper0_suction_cup_visual")
        if geom_id is None:
            geom_id = self._find_geom_id_by_substring("suction_cup_collision")

        self._suction_geom_id = geom_id

        body_id = self._safe_body_name2id("gripper0_suction_cup")
        if body_id is None and geom_id is not None:
            body_id = int(self.sim.model.geom_bodyid[geom_id])
        self._suction_body_id = body_id

        self._suction_site_id = self._safe_site_name2id("gripper0_suction_cup_site")
        self._suction_eq_id = self._safe_eq_name2id("gripper0_suction_runtime_weld")
        if self._suction_eq_id is None:
            self._suction_eq_id = self._find_eq_id_by_substring("suction_runtime_weld")

        self._resolve_suction_sensor_ids()

    def _resolve_suction_sensor_ids(self):
        if self._suction_force_sensor_id is not None and self._suction_torque_sensor_id is not None:
            return

        self._suction_force_sensor_id = self._safe_sensor_name2id("gripper0_suction_force")
        self._suction_torque_sensor_id = self._safe_sensor_name2id("gripper0_suction_torque")

        if self._suction_force_sensor_id is None:
            self._suction_force_sensor_id = self._find_sensor_id_by_substring("suction_force")
        if self._suction_torque_sensor_id is None:
            self._suction_torque_sensor_id = self._find_sensor_id_by_substring("suction_torque")

    def _get_body_subtree_body_ids(self, root_body_id):
        """
        返回一个 body 的整棵子树 body id 列表。

        这里用 MuJoCo 的 body_parentid 手动做一次遍历，避免依赖外部工具函数。
        对当前吸盘 gripper 而言，这棵子树通常包括：
        - suction_cup
        - eef

        注意：
        - 通过 equality weld 吸上的物体不会自动进入 body 亲子树；
        - 因此这里天然只会统计“吸盘自身工具子树”，不会把被吸住物体误算成自重。
        """
        root_body_id = int(root_body_id)
        parent_ids = np.asarray(self.sim.model.body_parentid, dtype=np.int64)
        body_ids = []

        for body_id in range(len(parent_ids)):
            current = int(body_id)
            while current >= 0:
                if current == root_body_id:
                    body_ids.append(int(body_id))
                    break

                parent = int(parent_ids[current])
                if parent == current:
                    break
                current = parent

        return tuple(body_ids)

    def _resolve_suction_tool_body_ids(self):
        """缓存吸盘工具子树的 body ids。"""
        if self._suction_tool_body_ids is not None:
            return self._suction_tool_body_ids

        self._resolve_suction_handles()
        if self._suction_body_id is None:
            self._suction_tool_body_ids = tuple()
            return self._suction_tool_body_ids

        self._suction_tool_body_ids = self._get_body_subtree_body_ids(self._suction_body_id)
        return self._suction_tool_body_ids

    def _get_suction_tool_mass_and_com_world(self):
        """
        计算吸盘工具子树的总质量和世界坐标系下质心。

        这里使用 body_mass + data.xipos：
        - body_mass：每个 body 的质量
        - xipos：每个 body 的世界坐标系下质心位置

        这比简单取 body_xpos 更合理，因为我们真正需要的是重力作用点。
        """
        body_ids = self._resolve_suction_tool_body_ids()
        if len(body_ids) == 0:
            return 0.0, np.zeros(3, dtype=np.float64)

        masses = np.asarray(self.sim.model.body_mass[list(body_ids)], dtype=np.float64)
        total_mass = float(np.sum(masses))
        if total_mass <= 1e-12:
            return 0.0, np.zeros(3, dtype=np.float64)

        # xipos 是 body 的世界坐标系质心位置。
        com_positions = np.asarray(self.sim.data.xipos[list(body_ids)], dtype=np.float64)
        weighted_com = np.sum(com_positions * masses[:, None], axis=0) / total_mass
        return total_mass, weighted_com

    def _get_body_subtree_mass_and_com_world(self, root_body_id):
        """
        计算某个 body 子树的总质量和世界坐标系质心。

        输入：
        - root_body_id：被吸附物体的根 body id。

        输出：
        - total_mass：这个物体子树的总质量。
        - weighted_com：这个物体子树的世界坐标系质心。

        这里只用质量和当前位置，不用速度、加速度或约束力。
        因此它适合用来生成“排除加速度干扰”的准静态重力 wrench。
        """
        if root_body_id is None:
            return 0.0, np.zeros(3, dtype=np.float64)

        body_ids = self._get_body_subtree_body_ids(int(root_body_id))
        if len(body_ids) == 0:
            return 0.0, np.zeros(3, dtype=np.float64)

        masses = np.asarray(self.sim.model.body_mass[list(body_ids)], dtype=np.float64)
        total_mass = float(np.sum(masses))
        if total_mass <= 1e-12:
            return 0.0, np.zeros(3, dtype=np.float64)

        com_positions = np.asarray(self.sim.data.xipos[list(body_ids)], dtype=np.float64)
        weighted_com = np.sum(com_positions * masses[:, None], axis=0) / total_mass
        return total_mass, weighted_com

    def _get_latched_object_gravity_wrench_world(self):
        """
        计算锁附物体对吸盘造成的准静态重力 wrench。

        返回值含义：
        - 前 3 维是被吸附物体的重力向量；
        - 后 3 维是以 suction_cup_site 为参考点的重力力矩 r x F。

        这里故意不读取 raw 传感器、速度、加速度或 weld 约束力。
        这样 payload 重力项不会被机械臂运动加速度污染。
        """
        if not self._suction_latched or self._latched_body_id is None:
            return np.zeros(6, dtype=np.float64)

        site_pos_world, _ = self._get_suction_site_world_pose()
        object_mass, object_com_world = self._get_body_subtree_mass_and_com_world(self._latched_body_id)
        if object_mass <= 1e-12:
            return np.zeros(6, dtype=np.float64)

        gravity_world = np.asarray(self.sim.model.opt.gravity[:3], dtype=np.float64)
        gravity_force_world = float(object_mass) * gravity_world
        lever_arm_world = np.asarray(object_com_world - site_pos_world, dtype=np.float64)
        gravity_torque_world = np.cross(lever_arm_world, gravity_force_world)
        return np.concatenate([gravity_force_world, gravity_torque_world], axis=0)

    def _get_payload_body_ids(self):
        """
        返回当前由吸盘承载的 payload body ids。

        当前仿真里 payload 由 suction weld 的锁附对象决定。
        没有锁附对象时，payload 集合为空；主显示公式仍然保持一致。
        """
        if not self._suction_latched or self._latched_body_id is None:
            return tuple()
        return self._get_body_subtree_body_ids(int(self._latched_body_id))

    def _compute_contact_wrench_on_body_set_world(self, contact_id, contact, body_set_is_geom1):
        """
        计算某个接触点中“外界作用在承载系统上”的世界系 wrench。

        输入：
        - body_set_is_geom1：True 表示 contact.geom1 属于承载系统；
          False 表示 contact.geom2 属于承载系统。

        输出：
        - 以 suction_cup_site 为参考点的 6D wrench。
        """
        contact_wrench_local = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(
            self.sim.model._model,
            self.sim.data._data,
            int(contact_id),
            contact_wrench_local,
        )

        rot_contact_to_world = self._contact_frame_rows_to_world_rot(contact)
        force_world = rot_contact_to_world.dot(np.asarray(contact_wrench_local[:3], dtype=np.float64))
        torque_world_at_contact = rot_contact_to_world.dot(np.asarray(contact_wrench_local[3:], dtype=np.float64))

        # MuJoCo contact force 可理解为“geom1 作用在 geom2 上”。
        # 如果承载系统是 geom1，就取反得到外界 geom2 作用在承载系统上的力；
        # 如果承载系统是 geom2，就直接使用 geom1 作用在 geom2 上的力。
        sign = -1.0 if body_set_is_geom1 else 1.0
        force_world *= sign
        torque_world_at_contact *= sign

        contact_pos_world = np.asarray(contact.pos[:3], dtype=np.float64)
        site_pos_world, _ = self._get_suction_site_world_pose()
        lever_arm_world = contact_pos_world - site_pos_world
        torque_world_at_site = torque_world_at_contact + np.cross(lever_arm_world, force_world)
        return np.concatenate([force_world, torque_world_at_site], axis=0)

    def _get_quasistatic_external_contact_wrench_world(self):
        """
        统一汇总吸盘承载系统受到的外部接触 wrench。

        承载系统 = 吸盘工具子树 + payload 子树。
        只统计承载系统和外界之间的接触；承载系统内部接触不计入主显示。
        这样可以保留 cup 压物体、payload 压桌面等压力，同时排除吸盘和 payload 之间的内部连接力。
        """
        self._resolve_suction_handles()

        tool_body_ids = set(self._resolve_suction_tool_body_ids())
        payload_body_ids = set(self._get_payload_body_ids())
        carrier_body_ids = tool_body_ids | payload_body_ids
        if len(carrier_body_ids) == 0:
            return np.zeros(6, dtype=np.float64)

        total_wrench_world = np.zeros(6, dtype=np.float64)
        for contact_id in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.sim.model.geom_bodyid[geom1])
            body2 = int(self.sim.model.geom_bodyid[geom2])

            geom1_in_carrier = body1 in carrier_body_ids
            geom2_in_carrier = body2 in carrier_body_ids

            # 两边同属承载系统是内部接触；两边都不属于承载系统则和吸盘端无关。
            if geom1_in_carrier == geom2_in_carrier:
                continue

            total_wrench_world += self._compute_contact_wrench_on_body_set_world(
                contact_id=contact_id,
                contact=contact,
                body_set_is_geom1=geom1_in_carrier,
            )

        return total_wrench_world

    def _get_suction_site_world_pose(self):
        """返回吸盘 site 的世界坐标系位置和旋转矩阵。"""
        self._resolve_suction_handles()

        if self._suction_site_id is None:
            return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)

        site_id = int(self._suction_site_id)
        site_pos = np.asarray(self.sim.data.site_xpos[site_id], dtype=np.float64).copy()
        site_rot = np.asarray(self.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
        return site_pos, site_rot

    def _get_suction_wrench_local_raw(self):
        """
        读取 MuJoCo 原始局部坐标系 wrench。

        这里返回的仍然是挂在 suction_cup_site 上的 raw force / torque，
        没做任何世界系变换、重力补偿或零偏补偿。
        """
        self._resolve_suction_sensor_ids()

        if self._suction_force_sensor_id is None or self._suction_torque_sensor_id is None:
            return None

        f_adr, f_dim = self._get_sensor_adr_dim(self._suction_force_sensor_id)
        t_adr, t_dim = self._get_sensor_adr_dim(self._suction_torque_sensor_id)
        if f_dim != 3 or t_dim != 3:
            return None

        force = np.array(self.sim.data.sensordata[f_adr : f_adr + 3], dtype=np.float64)
        torque = np.array(self.sim.data.sensordata[t_adr : t_adr + 3], dtype=np.float64)
        return np.concatenate([force, torque], axis=0)

    def _local_wrench_to_world(self, wrench_local):
        """把局部坐标系 wrench 旋转到世界坐标系。"""
        if wrench_local is None:
            return None

        _, site_rot = self._get_suction_site_world_pose()
        force_world = site_rot.dot(np.asarray(wrench_local[:3], dtype=np.float64))
        torque_world = site_rot.dot(np.asarray(wrench_local[3:], dtype=np.float64))
        return np.concatenate([force_world, torque_world], axis=0)

    def _get_tool_gravity_wrench_world(self):
        """
        计算吸盘工具子树在世界坐标系下的重力 wrench。

        这里“工具子树”只包含吸盘自身，不包含通过 equality weld 暂时吸住的物体。
        因此后续不会把被吸住物体的重量错误减掉。
        """
        self._resolve_suction_handles()
        site_pos_world, _ = self._get_suction_site_world_pose()
        total_mass, tool_com_world = self._get_suction_tool_mass_and_com_world()
        self._suction_tool_mass = float(total_mass)

        if total_mass <= 1e-12:
            return np.zeros(6, dtype=np.float64)

        gravity_world = np.asarray(self.sim.model.opt.gravity[:3], dtype=np.float64)
        gravity_force_world = total_mass * gravity_world
        lever_arm_world = np.asarray(tool_com_world - site_pos_world, dtype=np.float64)
        gravity_torque_world = np.cross(lever_arm_world, gravity_force_world)
        return np.concatenate([gravity_force_world, gravity_torque_world], axis=0)

    def _is_suction_bias_calibration_state(self):
        """
        判断当前是否适合做“空载静止零偏校准”。

        只有在下面这些条件都满足时，才更新 bias：
        - 吸盘未开启
        - 未锁附任何物体
        - 当前没有 cup 接触
        - 吸盘本体线速度和角速度都很小
        """
        self._resolve_suction_handles()

        if self._suction_enabled or self._suction_latched:
            return False
        if bool(self._suction_debug.get("cup_contact", False)):
            return False
        if self._suction_body_id is None:
            return False

        body_id = int(self._suction_body_id)
        body_cvel = np.asarray(self.sim.data.cvel[body_id], dtype=np.float64)
        angular_speed = float(np.linalg.norm(body_cvel[:3]))
        linear_speed = float(np.linalg.norm(body_cvel[3:6]))

        return angular_speed < 0.05 and linear_speed < 0.02

    def _resolve_suction_gravity_sign(self, raw_world, gravity_world):
        """
        自动判定 raw wrench 与理论重力 wrench 的符号关系。

        MuJoCo 站点力传感器的正负号，直觉上容易搞反。
        因此这里不硬编码符号，而是在“空载静止”状态下比较：
        - raw - gravity
        - raw + gravity
        哪个更接近 0，就选哪一个符号。
        """
        if self._suction_gravity_sign is not None:
            return float(self._suction_gravity_sign)

        plus_residual = np.linalg.norm(raw_world - gravity_world)
        minus_residual = np.linalg.norm(raw_world + gravity_world)
        self._suction_gravity_sign = 1.0 if plus_residual <= minus_residual else -1.0
        return float(self._suction_gravity_sign)

    def _update_suction_wrench_cache(self, update_bias=True):
        """
        更新当前时刻的 suction wrench 各种表示形式。

        这里同时维护三类信息：
        1. raw / world / gravity-compensated / bias-compensated 总载荷量
           这类值仍然保留，作为调试信息使用
        2. 接触瞬时外力（contact-based wrench）
           这类值保留为 cup 直接接触调试项
        3. 主显示 wrench（display wrench）
           统一等于“承载系统外部接触项 + payload 重力项”
        """
        local_raw = self._get_suction_wrench_local_raw()
        if local_raw is None:
            self._suction_wrench_local_raw = np.zeros(6, dtype=np.float64)
            self._suction_wrench_world_raw = np.zeros(6, dtype=np.float64)
            self._suction_wrench_gravity_world = np.zeros(6, dtype=np.float64)
            self._suction_wrench_compensated = np.zeros(6, dtype=np.float64)
        else:
            world_raw = self._local_wrench_to_world(local_raw)
            gravity_wrench_world = self._get_tool_gravity_wrench_world()

            gravity_sign = self._resolve_suction_gravity_sign(world_raw, gravity_wrench_world)
            gravity_contrib_world = gravity_sign * gravity_wrench_world
            residual_world = world_raw - gravity_contrib_world

            if update_bias and self._is_suction_bias_calibration_state():
                if not self._suction_bias_initialized:
                    self._suction_bias_world = residual_world.copy()
                    self._suction_bias_initialized = True
                else:
                    alpha = float(self._suction_bias_alpha)
                    self._suction_bias_world = (1.0 - alpha) * self._suction_bias_world + alpha * residual_world

            compensated_world = residual_world - self._suction_bias_world

            self._suction_wrench_local_raw = np.asarray(local_raw, dtype=np.float64).copy()
            self._suction_wrench_world_raw = np.asarray(world_raw, dtype=np.float64).copy()
            self._suction_wrench_gravity_world = np.asarray(gravity_contrib_world, dtype=np.float64).copy()
            self._suction_wrench_compensated = np.asarray(compensated_world, dtype=np.float64).copy()

        self._suction_contact_wrench_world = np.asarray(
            self._get_suction_contact_wrench_world(), dtype=np.float64
        ).copy()
        self._suction_latched_gravity_wrench_world = np.asarray(
            self._get_latched_object_gravity_wrench_world(), dtype=np.float64
        ).copy()
        self._suction_external_contact_wrench_world = np.asarray(
            self._get_quasistatic_external_contact_wrench_world(), dtype=np.float64
        ).copy()
        self._suction_payload_gravity_wrench_world = self._suction_latched_gravity_wrench_world.copy()
        self._suction_payload_quasistatic_wrench_world = (
            self._suction_external_contact_wrench_world + self._suction_payload_gravity_wrench_world
        )
        # 主显示全程使用同一个公式，不再按锁附前后切换口径。
        # 公式不使用速度、加速度或 weld 约束力，因此不会出现加速度干扰。
        self._suction_display_wrench_world = self._suction_payload_quasistatic_wrench_world.copy()

    def _safe_eq_name2id(self, name):
        raw_model = self.sim.model._model
        eq_id = int(mujoco.mj_name2id(raw_model, mujoco.mjtObj.mjOBJ_EQUALITY, name))
        if eq_id < 0:
            return None
        return eq_id

    def _find_eq_id_by_substring(self, token):
        raw_model = self.sim.model._model
        for eq_id in range(int(raw_model.neq)):
            eq_name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id)
            if eq_name is not None and token in eq_name:
                return eq_id
        return None

    def _get_sensor_adr_dim(self, sensor_id):
        adr = int(self.sim.model.sensor_adr[sensor_id])
        dim = int(self.sim.model.sensor_dim[sensor_id])
        return adr, dim

    def _contact_frame_rows_to_world_rot(self, contact):
        """
        把 MuJoCo contact.frame 转成“接触系 -> 世界系”的旋转矩阵。

        MuJoCo 文档里说明：
        - contact.frame 以转置形式存储
        - reshape(3, 3) 后，每一行是一个接触坐标轴在世界系下的表示

        因此：
        - 若 local_force 是接触坐标系下的向量
        - 则 world_force = frame_rows.T @ local_force
        """
        frame_rows = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        return frame_rows.T

    def _compute_contact_wrench_on_cup_world(self, contact_id, contact, cup_is_geom1):
        """
        计算某一个接触点“外界施加在吸盘 cup 上”的世界系 6D wrench。

        实现步骤：
        1. 用 mj_contactForce 读取接触坐标系下的 6D wrench
        2. 旋转到世界坐标系
        3. 根据 cup 是 geom1 还是 geom2，统一成“作用在 cup 上”的方向
        4. 以 suction_cup_site 为参考点，补上 r x f 形成总力矩
        """
        contact_wrench_local = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(
            self.sim.model._model,
            self.sim.data._data,
            int(contact_id),
            contact_wrench_local,
        )

        rot_contact_to_world = self._contact_frame_rows_to_world_rot(contact)
        force_world = rot_contact_to_world.dot(np.asarray(contact_wrench_local[:3], dtype=np.float64))
        torque_world_at_contact = rot_contact_to_world.dot(np.asarray(contact_wrench_local[3:], dtype=np.float64))

        # MuJoCo 的 contact force 方向可理解为“geom1 作用在 geom2 上”的接触 wrench。
        # 我们要统一成“外界作用在 suction cup 上”的方向：
        # - 如果 cup 是 geom1，则 cup 受到的是反方向
        # - 如果 cup 是 geom2，则 cup 受到的是正方向
        sign = -1.0 if cup_is_geom1 else 1.0
        force_world *= sign
        torque_world_at_contact *= sign

        contact_pos_world = np.asarray(contact.pos[:3], dtype=np.float64)
        site_pos_world, _ = self._get_suction_site_world_pose()
        lever_arm_world = contact_pos_world - site_pos_world
        torque_world_at_site = torque_world_at_contact + np.cross(lever_arm_world, force_world)

        return np.concatenate([force_world, torque_world_at_site], axis=0)

    def _get_suction_contact_wrench_world(self):
        """
        汇总所有“吸盘 cup 参与的接触点”，得到世界坐标系下的瞬时接触外力 / 外矩。

        这是接触未锁附阶段的主来源：
        - 有接触时立刻响应
        - 脱离接触后尽快回到接近 0
        """
        self._resolve_suction_handles()

        if self._suction_geom_id is None:
            return np.zeros(6, dtype=np.float64)

        total_wrench_world = np.zeros(6, dtype=np.float64)
        for contact_id in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if geom1 != self._suction_geom_id and geom2 != self._suction_geom_id:
                continue

            cup_is_geom1 = geom1 == self._suction_geom_id
            total_wrench_world += self._compute_contact_wrench_on_cup_world(
                contact_id=contact_id,
                contact=contact,
                cup_is_geom1=cup_is_geom1,
            )

        return total_wrench_world

    def _get_suction_wrench(self):
        """
        返回对外暴露的主语义 wrench。

        当前主语义是给屏幕显示和 obs 使用的统一准静态外载荷：
        - 外部接触力会进入示数；
        - payload 重力会进入示数；
        - 吸盘自身重力、内部连接力、速度和加速度不会进入主示数。
        """
        self._update_suction_wrench_cache(update_bias=False)
        return np.asarray(self._suction_display_wrench_world, dtype=np.float64).copy()

    def _resolve_force_body_ids(self):
        if self._force_body_ids is not None:
            return self._force_body_ids

        left_id = self._safe_body_name2id("gripper0_finger_joint1_tip")
        right_id = self._safe_body_name2id("gripper0_finger_joint2_tip")

        if left_id is None or right_id is None:
            self._resolve_suction_handles()
            if self._suction_body_id is not None:
                if left_id is None:
                    left_id = self._suction_body_id
                if right_id is None:
                    right_id = self._suction_body_id

        if left_id is None and right_id is None:
            left_id = right_id = 0
        elif left_id is None:
            left_id = right_id
        elif right_id is None:
            right_id = left_id

        self._force_body_ids = (int(left_id), int(right_id))
        return self._force_body_ids

    def _find_free_joint_ancestor(self, body_id):
        current_body = int(body_id)
        while current_body != -1:
            joint_num = int(self.sim.model.body_jntnum[current_body])
            joint_adr = int(self.sim.model.body_jntadr[current_body])

            if joint_num > 0 and joint_adr >= 0:
                joint_id = joint_adr
                if self.sim.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                    return current_body, joint_id
                return None, None

            parent = int(self.sim.model.body_parentid[current_body])
            if parent == current_body:
                break
            current_body = parent

        return None, None

    def _set_weld_active(self, eq_id, active):
        if eq_id is None:
            return

        if hasattr(self.sim.data, "eq_active"):
            self.sim.data.eq_active[eq_id] = 1 if active else 0
        elif hasattr(self.sim.model, "eq_active"):
            self.sim.model.eq_active[eq_id] = 1 if active else 0

    def _is_weld_active(self, eq_id):
        if eq_id is None:
            return False

        if hasattr(self.sim.data, "eq_active"):
            return bool(self.sim.data.eq_active[eq_id])
        if hasattr(self.sim.model, "eq_active"):
            return bool(self.sim.model.eq_active[eq_id])
        return False

    def _compute_contact_normal(self, contact, cup_is_geom1):
        frame = np.array(contact.frame).reshape(3, 3)
        normal = frame[:, 0].copy()
        if not cup_is_geom1:
            normal = -normal

        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return normal / norm

    def _get_body_linear_velocity(self, body_id):
        # cvel: [ang_vel(3), lin_vel(3)] in world frame
        return np.array(self.sim.data.cvel[int(body_id)][3:6], dtype=np.float64)

    def _get_relative_linear_speed(self, body_a, body_b):
        vel_a = self._get_body_linear_velocity(body_a)
        vel_b = self._get_body_linear_velocity(body_b)
        return float(np.linalg.norm(vel_a - vel_b))

    def _get_suction_overlap_sample_offsets(self):
        # 这个函数生成吸盘圆面上的固定采样点。
        # 输入是当前吸盘半径，输出是二维圆盘坐标点。
        # 固定采样点可以避免每一步随机变化导致吸附判断抖动。
        if self._suction_overlap_sample_offsets is not None:
            return self._suction_overlap_sample_offsets

        # count 表示圆盘上采样点数量，数量越多越稳定，但每步计算越慢。
        count = max(1, int(self._suction_overlap_sample_count))
        # golden_angle 是黄金角，用它绕圆盘撒点可以比普通网格更均匀。
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        # offsets 用来保存单位圆盘里的 (x, y) 偏移，真正半径会在使用时再乘进去。
        offsets = []
        for index in range(count):
            # sqrt 让采样点按面积均匀分布，而不是挤在圆心附近。
            radius = np.sqrt((float(index) + 0.5) / float(count))
            # angle 控制当前采样点绕圆心旋转到哪个方向。
            angle = float(index) * golden_angle
            # 这里存的是单位圆盘坐标，范围大致在 [-1, 1]。
            offsets.append([radius * np.cos(angle), radius * np.sin(angle)])

        self._suction_overlap_sample_offsets = np.asarray(offsets, dtype=np.float64)
        return self._suction_overlap_sample_offsets

    def _get_suction_disk_frame(self):
        # 这个函数读取吸盘 collision geom 的世界位姿。
        # 输出是圆盘中心、两个圆盘平面轴和半径。
        # 吸盘 collision 在 XML 里是 cylinder，局部 x/y 轴张成圆面，size[0] 是半径。
        self._resolve_suction_handles()

        if self._suction_geom_id is None:
            return None, None, None, 0.0

        geom_id = int(self._suction_geom_id)
        # geom_xpos 是吸盘 collision geom 中心在世界坐标系的位置。
        disk_center = np.asarray(self.sim.data.geom_xpos[geom_id], dtype=np.float64).copy()
        # geom_xmat 是局部坐标到世界坐标的旋转矩阵。
        # reshape 后每一列对应一个局部轴在世界系的方向。
        disk_rot = np.asarray(self.sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3).copy()
        # 局部 x/y 轴就是圆盘平面内的两个方向。
        disk_axis_x = disk_rot[:, 0].copy()
        disk_axis_y = disk_rot[:, 1].copy()
        # geom_size 的第一个数是 cylinder 半径。
        # 如果异常就退回到 suction_gripper.xml 里的默认 0.020m。
        radius = float(self.sim.model.geom_size[geom_id][0])
        if radius <= 1e-8:
            radius = 0.020

        return disk_center, disk_axis_x, disk_axis_y, radius

    def _orient_contact_direction_from_cup_to_geom(self, contact, cup_is_geom1, other_geom):
        # 这个函数把 MuJoCo contact 法向统一成“从吸盘指向当前 other_geom”的方向。
        # 输入是当前 contact、吸盘是否为 geom1、被碰到的 other_geom；输出是单位方向向量。
        direction = self._compute_contact_normal(contact, cup_is_geom1=cup_is_geom1)

        # 用吸盘 geom 中心和 other_geom 中心做一次方向校正，避免 MuJoCo 法向符号和直觉相反。
        cup_pos = np.asarray(self.sim.data.geom_xpos[int(self._suction_geom_id)], dtype=np.float64)
        other_pos = np.asarray(self.sim.data.geom_xpos[int(other_geom)], dtype=np.float64)
        if float(np.dot(other_pos - cup_pos, direction)) < 0.0:
            direction = -direction

        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return direction / norm

    def _ray_to_geom_local(self, ray_origin_world, ray_dir_world, geom_id):
        # 这个函数把世界坐标系下的射线转换到目标 geom 的局部坐标系。
        # 这样后续可以直接对 box、cylinder 等标准几何体做求交。
        geom_id = int(geom_id)
        geom_pos = np.asarray(self.sim.data.geom_xpos[geom_id], dtype=np.float64)
        geom_rot = np.asarray(self.sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        ray_origin_local = geom_rot.T.dot(np.asarray(ray_origin_world, dtype=np.float64) - geom_pos)
        ray_dir_local = geom_rot.T.dot(np.asarray(ray_dir_world, dtype=np.float64))
        return ray_origin_local, ray_dir_local

    def _ray_box_hit_distance_local(self, ray_origin, ray_dir, half_size):
        # 这个函数计算局部坐标系下射线和 box 的最近命中距离。
        # half_size 是 box 的半长宽高，返回 None 表示没有命中。
        eps = 1e-9
        t_min = -np.inf
        t_max = np.inf

        for axis in range(3):
            origin_value = float(ray_origin[axis])
            direction_value = float(ray_dir[axis])
            limit = float(half_size[axis])

            # 如果射线几乎平行于当前轴对应的两张面，就只能检查起点是否在范围内。
            if abs(direction_value) < eps:
                if origin_value < -limit or origin_value > limit:
                    return None
                continue

            # 计算射线进入和离开当前轴 slab 的距离。
            t1 = (-limit - origin_value) / direction_value
            t2 = (limit - origin_value) / direction_value
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)

            if t_min > t_max:
                return None

        if t_max < 0.0:
            return None
        return max(float(t_min), 0.0)

    def _ray_ellipsoid_hit_distance_local(self, ray_origin, ray_dir, radii):
        # 这个函数计算局部坐标系下射线和 sphere/ellipsoid 的最近命中距离。
        # radii 是三个轴向半径；sphere 可以看成三个半径相同的 ellipsoid。
        radii = np.asarray(radii, dtype=np.float64)
        if np.any(radii <= 1e-8):
            return None

        scaled_origin = np.asarray(ray_origin, dtype=np.float64) / radii
        scaled_dir = np.asarray(ray_dir, dtype=np.float64) / radii
        a = float(np.dot(scaled_dir, scaled_dir))
        b = float(2.0 * np.dot(scaled_origin, scaled_dir))
        c = float(np.dot(scaled_origin, scaled_origin) - 1.0)

        if a <= 1e-12:
            return None
        if c <= 0.0:
            return 0.0

        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None

        sqrt_disc = float(np.sqrt(disc))
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        candidates = [t for t in (t1, t2) if t >= 0.0]
        if len(candidates) == 0:
            return None
        return float(min(candidates))

    def _ray_cylinder_hit_distance_local(self, ray_origin, ray_dir, radius, half_height):
        # 这个函数计算局部坐标系下射线和有限 cylinder 的最近命中距离。
        # MuJoCo cylinder 默认沿局部 z 轴，radius 是半径，half_height 是半高。
        eps = 1e-9
        radius = float(radius)
        half_height = float(half_height)
        if radius <= 1e-8 or half_height <= 1e-8:
            return None

        x, y, z = [float(v) for v in ray_origin]
        dx, dy, dz = [float(v) for v in ray_dir]

        # 如果射线起点已经在 cylinder 里面，说明这个采样点已经覆盖到目标 geom。
        if x * x + y * y <= radius * radius and abs(z) <= half_height:
            return 0.0

        hits = []
        # 先求侧面命中：x^2 + y^2 = r^2。
        a = dx * dx + dy * dy
        b = 2.0 * (x * dx + y * dy)
        c = x * x + y * y - radius * radius
        if a > eps:
            disc = b * b - 4.0 * a * c
            if disc >= 0.0:
                sqrt_disc = float(np.sqrt(disc))
                for t in ((-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)):
                    hit_z = z + t * dz
                    if t >= 0.0 and abs(hit_z) <= half_height:
                        hits.append(float(t))

        # 再求上下两个端面命中：z = +/- half_height。
        if abs(dz) > eps:
            for cap_z in (-half_height, half_height):
                t = (cap_z - z) / dz
                hit_x = x + t * dx
                hit_y = y + t * dy
                if t >= 0.0 and hit_x * hit_x + hit_y * hit_y <= radius * radius:
                    hits.append(float(t))

        if len(hits) == 0:
            return None
        return float(min(hits))

    def _ray_capsule_hit_distance_local(self, ray_origin, ray_dir, radius, half_length):
        # 这个函数计算局部坐标系下射线和 capsule 的最近命中距离。
        # capsule 可以近似看成一个 cylinder 加上下两个 sphere 端帽。
        radius = float(radius)
        half_length = float(half_length)
        if radius <= 1e-8:
            return None

        point = np.asarray(ray_origin, dtype=np.float64)
        direction = np.asarray(ray_dir, dtype=np.float64)
        closest_z = float(np.clip(point[2], -half_length, half_length))
        closest_on_axis = np.array([0.0, 0.0, closest_z], dtype=np.float64)
        if float(np.linalg.norm(point - closest_on_axis)) <= radius:
            return 0.0

        hits = []
        cylinder_hit = self._ray_cylinder_hit_distance_local(point, direction, radius, half_length)
        if cylinder_hit is not None:
            hits.append(float(cylinder_hit))

        for cap_z in (-half_length, half_length):
            cap_origin = point - np.array([0.0, 0.0, cap_z], dtype=np.float64)
            cap_hit = self._ray_ellipsoid_hit_distance_local(cap_origin, direction, [radius, radius, radius])
            if cap_hit is not None:
                hits.append(float(cap_hit))

        if len(hits) == 0:
            return None
        return float(min(hits))

    def _ray_hits_target_geom_with_mujoco(self, ray_origin_world, ray_dir_world, target_geom_id, max_dist):
        # 这个函数是复杂 geom 的兜底路径：用 MuJoCo 自己的 raycast 判断是否命中目标 geom。
        # 只有最近命中的 geom 正好等于 target_geom_id 时才算命中。
        # 这样可以避免把其它 geom 算进重合面积。
        geomid = np.array([-1], dtype=np.int32)
        bodyexclude = int(self._suction_body_id) if self._suction_body_id is not None else -1
        distance = mujoco.mj_ray(
            self.sim.model._model,
            self.sim.data._data,
            np.asarray(ray_origin_world, dtype=np.float64),
            np.asarray(ray_dir_world, dtype=np.float64),
            None,
            1,
            bodyexclude,
            geomid,
            None,
        )
        return bool(distance >= 0.0 and distance <= max_dist and int(geomid[0]) == int(target_geom_id))

    def _ray_hits_target_geom(self, ray_origin_world, ray_dir_world, target_geom_id, max_dist):
        # 这个函数只判断射线是否在 max_dist 内命中指定的 target_geom_id。
        # 对常见基础 geom 用解析求交，复杂 geom 再退回 MuJoCo raycast。
        hit_distance = self._ray_hit_distance_to_geom(
            ray_origin_world=ray_origin_world,
            ray_dir_world=ray_dir_world,
            target_geom_id=target_geom_id,
        )
        return bool(hit_distance is not None and hit_distance <= float(max_dist))

    def _ray_hit_distance_to_geom(self, ray_origin_world, ray_dir_world, target_geom_id):
        # 这个函数返回射线命中指定 geom 的最近距离。
        # 返回 None 表示没有命中；基础 geom 用解析求交，复杂 geom 用 MuJoCo 最近命中兜底。
        target_geom_id = int(target_geom_id)
        ray_origin_local, ray_dir_local = self._ray_to_geom_local(ray_origin_world, ray_dir_world, target_geom_id)
        geom_size = np.asarray(self.sim.model.geom_size[target_geom_id], dtype=np.float64)
        geom_type = int(self.sim.model.geom_type[target_geom_id])

        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            hit_distance = self._ray_box_hit_distance_local(ray_origin_local, ray_dir_local, geom_size[:3])
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            hit_distance = self._ray_cylinder_hit_distance_local(
                ray_origin_local, ray_dir_local, geom_size[0], geom_size[1]
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            hit_distance = self._ray_ellipsoid_hit_distance_local(
                ray_origin_local, ray_dir_local, [geom_size[0], geom_size[0], geom_size[0]]
            )
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            hit_distance = self._ray_ellipsoid_hit_distance_local(ray_origin_local, ray_dir_local, geom_size[:3])
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            hit_distance = self._ray_capsule_hit_distance_local(
                ray_origin_local, ray_dir_local, geom_size[0], geom_size[1]
            )
        else:
            geomid = np.array([-1], dtype=np.int32)
            bodyexclude = int(self._suction_body_id) if self._suction_body_id is not None else -1
            distance = mujoco.mj_ray(
                self.sim.model._model,
                self.sim.data._data,
                np.asarray(ray_origin_world, dtype=np.float64),
                np.asarray(ray_dir_world, dtype=np.float64),
                None,
                1,
                bodyexclude,
                geomid,
                None,
            )
            if distance >= 0.0 and int(geomid[0]) == target_geom_id:
                return float(distance)
            return None

        return hit_distance

    def _is_collision_geom(self, geom_id):
        # 这个函数过滤掉纯 visual geom，避免可视外壳误参与吸附面积统计。
        geom_id = int(geom_id)
        contype = int(self.sim.model.geom_contype[geom_id])
        conaffinity = int(self.sim.model.geom_conaffinity[geom_id])
        return bool(contype != 0 or conaffinity != 0)

    def _get_candidate_collision_geom_ids(self, candidate_body_id):
        # 这个函数收集当前候选物体名下的所有 collision geom，供 96 条采样射线复用。
        candidate_body_id = int(candidate_body_id)
        candidate_geom_ids = []

        for geom_id in range(len(self.sim.model.geom_bodyid)):
            if not self._is_collision_geom(geom_id):
                continue

            geom_body_id = int(self.sim.model.geom_bodyid[geom_id])
            hit_body_id, _ = self._find_free_joint_ancestor(geom_body_id)
            if hit_body_id == candidate_body_id:
                candidate_geom_ids.append(int(geom_id))

        return candidate_geom_ids

    def _find_candidate_body_ray_hit(self, ray_origin_world, ray_dir_world, candidate_geom_ids, max_dist):
        # 这个函数检查射线是否命中当前候选物体的任意 collision geom。
        # 输入是射线、候选物体 geom 列表和最大距离；输出是 (是否命中, 命中距离, 命中 geom)。
        best_distance = None
        best_geom_id = -1

        for geom_id in candidate_geom_ids:
            hit_distance = self._ray_hit_distance_to_geom(
                ray_origin_world=ray_origin_world,
                ray_dir_world=ray_dir_world,
                target_geom_id=geom_id,
            )
            if hit_distance is None or hit_distance > float(max_dist):
                continue
            if best_distance is None or hit_distance < best_distance:
                best_distance = float(hit_distance)
                best_geom_id = int(geom_id)

        if best_distance is None:
            return False, float(max_dist), -1
        return True, float(best_distance), int(best_geom_id)

    def _estimate_suction_projection_overlap_ratio(self, candidate_body_id):
        # 这个函数估算“吸盘圆盘竖直投影到当前候选物体上的重合比例”。
        # 输入是当前候选物体 free-joint 根 body，输出是 0 到 1 之间的命中比例。
        disk_center, _, _, disk_radius = self._get_suction_disk_frame()
        if disk_center is None or disk_radius <= 1e-8:
            return 0.0

        # 当前版本按用户需求固定使用世界坐标系 -Z 作为投影方向。
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        candidate_geom_ids = self._get_candidate_collision_geom_ids(candidate_body_id)
        if len(candidate_geom_ids) == 0:
            return 0.0

        # 全局 Z 轴作为吸盘圆面法向，所以采样圆盘落在世界 XY 平面里。
        disk_axis_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        disk_axis_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        # 采样点是单位圆盘坐标，这里乘上真实吸盘半径得到世界里的水平圆盘点。
        sample_offsets = self._get_suction_overlap_sample_offsets() * float(disk_radius)
        sample_count = int(len(sample_offsets))
        required_hits = int(np.ceil(float(sample_count) * float(self._suction_required_overlap_ratio)))
        hit_count = 0

        for index, (offset_x, offset_y) in enumerate(sample_offsets):
            # sample_world 是吸盘圆盘上的一个采样点。
            sample_world = disk_center + float(offset_x) * disk_axis_x + float(offset_y) * disk_axis_y
            # 射线从采样点稍微往吸盘外侧退一点再发出。
            # 这样可以减少数值穿插导致的起点已经在接触面内部的问题。
            ray_origin = sample_world - direction * float(self._suction_overlap_ray_start_offset)
            hit, _, _ = self._find_candidate_body_ray_hit(
                ray_origin_world=ray_origin,
                ray_dir_world=direction,
                candidate_geom_ids=candidate_geom_ids,
                max_dist=float(self._suction_overlap_projection_depth),
            )
            if hit:
                hit_count += 1

            # 早停 1：命中数已经达到 90% 门槛，继续算不会改变通过结果。
            if hit_count >= required_hits:
                return float(hit_count) / float(sample_count)

            # 早停 2：即使剩余采样点全命中，也不可能达到门槛，继续算不会改变失败结果。
            remaining_count = sample_count - int(index) - 1
            if hit_count + remaining_count < required_hits:
                return float(hit_count) / float(sample_count)

        return float(hit_count) / float(sample_count)

    def _compute_weld_relpose_from_current_pose(self, body_id):
        cup_pos = np.array(self.sim.data.body_xpos[self._suction_body_id])
        cup_quat = T.convert_quat(np.array(self.sim.data.body_xquat[self._suction_body_id]), to="xyzw")

        obj_pos = np.array(self.sim.data.body_xpos[body_id])
        obj_quat_xyzw = T.convert_quat(np.array(self.sim.data.body_xquat[body_id]), to="xyzw")

        T_cup = T.pose2mat((cup_pos, cup_quat))
        T_obj = T.pose2mat((obj_pos, obj_quat_xyzw))
        T_rel = T.pose_inv(T_cup).dot(T_obj)
        rel_pos, rel_quat_xyzw = T.mat2pose(T_rel)
        rel_quat_wxyz = T.convert_quat(np.array(rel_quat_xyzw), to="wxyz")
        return np.array(rel_pos), np.array(rel_quat_wxyz)

    def _compute_weld_relpose_soft_contact(self, body_id, contact, cup_is_geom1, pull_dist=0.0005):
        cup_pos = np.array(self.sim.data.body_xpos[self._suction_body_id])
        cup_quat = T.convert_quat(np.array(self.sim.data.body_xquat[self._suction_body_id]), to="xyzw")

        obj_pos = np.array(self.sim.data.body_xpos[body_id])
        obj_quat_xyzw = T.convert_quat(np.array(self.sim.data.body_xquat[body_id]), to="xyzw")

        normal = self._compute_contact_normal(contact, cup_is_geom1=cup_is_geom1)
        obj_pos_soft = obj_pos - normal * float(pull_dist)

        T_cup = T.pose2mat((cup_pos, cup_quat))
        T_obj = T.pose2mat((obj_pos_soft, obj_quat_xyzw))
        T_rel = T.pose_inv(T_cup).dot(T_obj)
        rel_pos, rel_quat_xyzw = T.mat2pose(T_rel)
        rel_quat_wxyz = T.convert_quat(np.array(rel_quat_xyzw), to="wxyz")
        return np.array(rel_pos), np.array(rel_quat_wxyz)

    def _bind_suction_weld(self, body_id, rel_pos, rel_quat_wxyz):
        if self._suction_eq_id is None or self._suction_body_id is None:
            return False

        eq_id = int(self._suction_eq_id)

        self.sim.model.eq_obj1id[eq_id] = int(self._suction_body_id)
        self.sim.model.eq_obj2id[eq_id] = int(body_id)

        # eq_data for mjEQ_WELD:
        # [0:3] anchor, [3:6] relpos, [6:10] relquat, [10] torquescale
        self.sim.model.eq_data[eq_id, 0:3] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.sim.model.eq_data[eq_id, 3:6] = rel_pos
        self.sim.model.eq_data[eq_id, 6:10] = rel_quat_wxyz

        self._set_weld_active(eq_id, True)
        mujoco.mj_forward(self.sim.model._model, self.sim.data._data)
        return True

    def _start_suction_latch(self, body_id, joint_id, contact=None, cup_is_geom1=True):
        if self._suction_body_id is None:
            return

        if contact is None:
            rel_pos, rel_quat_wxyz = self._compute_weld_relpose_from_current_pose(
                body_id=body_id,
            )
        else:
            rel_pos, rel_quat_wxyz = self._compute_weld_relpose_soft_contact(
                body_id=body_id,
                contact=contact,
                cup_is_geom1=cup_is_geom1,
                pull_dist=0.0005,
            )

        if not self._bind_suction_weld(body_id=body_id, rel_pos=rel_pos, rel_quat_wxyz=rel_quat_wxyz):
            return

        self._suction_latched = True
        self._latched_body_id = int(body_id)
        self._latched_joint_id = int(joint_id)

    def _maintain_suction_latch(self):
        if not self._suction_latched:
            return

        if self.sim.model.jnt_type[self._latched_joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            self._release_suction_latch()
            return

        if not self._is_weld_active(self._suction_eq_id):
            self._release_suction_latch()
            return
            
        # ---------- Simulate Vacuum Seal Break (Leakage on heavy force) ----------
        cup_pos = np.array(self.sim.data.body_xpos[self._suction_body_id])
        cup_quat = T.convert_quat(np.array(self.sim.data.body_xquat[self._suction_body_id]), to="xyzw")
        
        obj_pos = np.array(self.sim.data.body_xpos[self._latched_body_id])
        obj_quat_xyzw = T.convert_quat(np.array(self.sim.data.body_xquat[self._latched_body_id]), to="xyzw")

        T_cup = T.pose2mat((cup_pos, cup_quat))
        T_obj = T.pose2mat((obj_pos, obj_quat_xyzw))
        T_rel = T.pose_inv(T_cup).dot(T_obj)
        current_rel_pos, current_rel_quat_xyzw = T.mat2pose(T_rel)
        
        eq_id = int(self._suction_eq_id)
        target_rel_pos = np.array(self.sim.model.eq_data[eq_id, 3:6])
        target_rel_quat_wxyz = np.array(self.sim.model.eq_data[eq_id, 6:10])
        target_rel_quat_xyzw = T.convert_quat(target_rel_quat_wxyz, to="xyzw")

        pos_error = float(np.linalg.norm(current_rel_pos - target_rel_pos))
        dot_product = float(np.clip(np.abs(np.dot(current_rel_quat_xyzw, target_rel_quat_xyzw)), -1.0, 1.0))
        angle_error = float(2 * np.arccos(dot_product))

        # Thresholds: ~2.5cm translation or ~25 degrees bending means the seal is broken.
        if pos_error > 0.025 or angle_error > 0.436:  # 0.436 rad is roughly 25 degrees
            # print(f"[Suction Sim] Seal broken due to large external force! pos_err: {pos_error*100:.1f}cm, angle_err: {np.degrees(angle_error):.1f}deg")
            self._release_suction_latch()
            return
            
        # ---------- Simulate Vacuum Seal Break (Leakage on heavy force) ----------
        cup_pos = np.array(self.sim.data.body_xpos[self._suction_body_id])
        cup_quat = T.convert_quat(np.array(self.sim.data.body_xquat[self._suction_body_id]), to="xyzw")
        
        obj_pos = np.array(self.sim.data.body_xpos[self._latched_body_id])
        obj_quat_xyzw = T.convert_quat(np.array(self.sim.data.body_xquat[self._latched_body_id]), to="xyzw")

        T_cup = T.pose2mat((cup_pos, cup_quat))
        T_obj = T.pose2mat((obj_pos, obj_quat_xyzw))
        T_rel = T.pose_inv(T_cup).dot(T_obj)
        current_rel_pos, current_rel_quat_xyzw = T.mat2pose(T_rel)
        
        eq_id = int(self._suction_eq_id)
        target_rel_pos = np.array(self.sim.model.eq_data[eq_id, 3:6])
        target_rel_quat_wxyz = np.array(self.sim.model.eq_data[eq_id, 6:10])
        target_rel_quat_xyzw = T.convert_quat(target_rel_quat_wxyz, to="xyzw")

        pos_error = float(np.linalg.norm(current_rel_pos - target_rel_pos))
        dot_product = float(np.clip(np.abs(np.dot(current_rel_quat_xyzw, target_rel_quat_xyzw)), -1.0, 1.0))
        angle_error = float(2 * np.arccos(dot_product))

        # Thresholds: ~2.5cm translation or ~25 degrees bending means the seal is broken.
        if pos_error > 0.025 or angle_error > 0.436:  # 0.436 rad is roughly 25 degrees
            # print(f"[Suction Sim] Seal broken due to large external force! pos_err: {pos_error*100:.1f}cm, angle_err: {np.degrees(angle_error):.1f}deg")
            self._release_suction_latch()

    def _release_suction_latch(self):
        self._set_weld_active(self._suction_eq_id, False)

        self._suction_latched = False
        self._latched_body_id = None
        self._latched_joint_id = None
        self._suction_candidate_body_id = None
        self._suction_candidate_joint_id = None
        self._suction_contact_stable_steps = 0

    def _try_latch_from_contacts(self):
        if self._suction_geom_id is None:
            return

        matched_body_id = None
        matched_joint_id = None
        matched_rel_speed = None
        matched_contact = None
        matched_cup_is_geom1 = True

        for i in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[i]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if geom1 != self._suction_geom_id and geom2 != self._suction_geom_id:
                continue

            self._suction_debug["cup_contact"] = True

            other_geom = geom2 if geom1 == self._suction_geom_id else geom1
            cup_is_geom1 = geom1 == self._suction_geom_id
            self._suction_debug["overlap_contact_geom_id"] = int(other_geom)
            other_body = int(self.sim.model.geom_bodyid[other_geom])
            body_id, joint_id = self._find_free_joint_ancestor(other_body)

            if joint_id is None:
                continue

            # 相对速度现在只作为调试信息记录，不再作为吸附门槛。
            # 这样吸盘开启但没有锁附成功时，不会因为尝试吸附而额外修改物体速度；
            # 真正决定是否吸附的条件只剩接触、投影重合比例和稳定步数。
            rel_speed = self._get_relative_linear_speed(self._suction_body_id, body_id)

            overlap_ratio = self._estimate_suction_projection_overlap_ratio(candidate_body_id=body_id)
            overlap_passed = overlap_ratio >= self._suction_required_overlap_ratio
            self._suction_debug["overlap_ratio"] = float(overlap_ratio)
            self._suction_debug["overlap_passed"] = bool(overlap_passed)
            if not overlap_passed:
                continue

            matched_body_id = int(body_id)
            matched_joint_id = int(joint_id)
            matched_rel_speed = float(rel_speed)
            matched_contact = contact
            matched_cup_is_geom1 = cup_is_geom1
            break

        if matched_body_id is None:
            self._suction_candidate_body_id = None
            self._suction_candidate_joint_id = None
            self._suction_contact_stable_steps = 0
            self._suction_debug["candidate_body_id"] = -1
            self._suction_debug["stable_steps"] = 0
            self._suction_debug["relative_speed"] = None
            return

        if self._suction_candidate_body_id == matched_body_id:
            self._suction_contact_stable_steps += 1
        else:
            self._suction_candidate_body_id = matched_body_id
            self._suction_candidate_joint_id = matched_joint_id
            self._suction_contact_stable_steps = 1

        self._suction_debug["candidate_body_id"] = int(self._suction_candidate_body_id)
        self._suction_debug["stable_steps"] = int(self._suction_contact_stable_steps)
        self._suction_debug["relative_speed"] = matched_rel_speed

        if self._suction_contact_stable_steps < self._suction_required_stable_steps:
            return

        self._start_suction_latch(
            body_id=self._suction_candidate_body_id,
            joint_id=self._suction_candidate_joint_id,
            contact=matched_contact,
            cup_is_geom1=matched_cup_is_geom1,
        )
        self._suction_debug["latched_joint_type"] = int(
            self.sim.model.jnt_type[self._suction_candidate_joint_id]
        )

    def _update_suction(self, action):
        self._resolve_suction_handles()

        self._suction_debug = {
            "ncon": int(self.sim.data.ncon),
            "cup_contact": False,
            "latched": bool(self._suction_latched),
            "latched_joint_type": None,
            "candidate_body_id": -1,
            "stable_steps": 0,
            "relative_speed": None,
            "overlap_ratio": -1.0,
            "overlap_passed": False,
            "overlap_contact_geom_id": -1,
        }

        cmd = float(action[6]) if len(action) > 6 else 0.0
        if cmd > 0.5:
            self._suction_enabled = True
        elif cmd < -0.5:
            self._suction_enabled = False

        self._suction_active = bool(self._suction_enabled)

        if not self._suction_active:
            self._release_suction_latch()
            self._suction_debug["latched"] = False
            return

        if not self._suction_latched:
            self._try_latch_from_contacts()
        else:
            self._maintain_suction_latch()

        self._suction_debug["latched"] = bool(self._suction_latched)

    def get_robot_state_vector(self, obs):
        return np.concatenate(
            [obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
        )

    def is_fixture(self, object_name):
        """
        Check if an object is defined as a fixture in the task

        Args:
            object_name (str): The name string of the object in query
        """
        return object_name in list(self.fixtures_dict.keys())

    @property
    def language_instruction(self):
        return self.parsed_problem["language"]

    def get_object(self, object_name):
        for query_dict in [
            self.fixtures_dict,
            self.objects_dict,
            self.object_sites_dict,
        ]:
            if object_name in query_dict:
                return query_dict[object_name]
