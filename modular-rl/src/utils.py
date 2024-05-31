from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xmltodict
import wrappers
import gym
from gym.envs.registration import register
from shutil import copyfile
from config import *


def sinkhorn(x, iterations=100):
    for _ in range(iterations):
        x = F.log_softmax(F.log_softmax(x, dim=-1), dim=-2)
    return x


def makeEnvWrapper(env_name, obs_max_len=None, seed=0):
    """return wrapped gym environment for parallel sample collection (vectorized environments)"""

    def helper():
        e = gym.make("environments:%s-v0" % env_name)
        e.seed(seed)
        return wrappers.ModularEnvWrapper(e, obs_max_len)

    return helper


def findMaxChildren(env_names, graphs):
    """return the maximum number of children given a list of env names and their corresponding graph structures"""
    max_children = 0
    for name in env_names:
        most_frequent = max(graphs[name], key=graphs[name].count)
        max_children = max(max_children, graphs[name].count(most_frequent))
    return max_children


def registerEnvs(env_names, max_episode_steps, custom_xml, use_restricted_obs=False):
    """register the MuJoCo envs with Gym and return the per-limb observation size and max action value (for modular policy training)"""
    # get all paths to xmls (handle the case where the given path is a directory containing multiple xml files)
    paths_to_register = []
    # existing envs
    if not custom_xml:
        for name in env_names:
            paths_to_register.append(os.path.join(XML_DIR, "{}.xml".format(name)))
    # custom envs
    else:
        if os.path.isfile(custom_xml):
            paths_to_register.append(custom_xml)
        elif os.path.isdir(custom_xml):
            for name in sorted(os.listdir(custom_xml)):
                if ".xml" in name:
                    paths_to_register.append(os.path.join(custom_xml, name))

    if not paths_to_register:
        raise ValueError(f"No XML files found to register environments in provided xml: {custom_xml}.")
    limb_obs_size = None
    max_action = None
    # register each env
    for xml in paths_to_register:
        env_name = os.path.basename(xml)[:-4]
        # create a copy of modular environment for custom xml model
        if not os.path.exists(os.path.join(ENV_DIR, f"{env_name}.py")):
            # create a duplicate of gym environment file for each env (necessary for avoiding bug in gym)
            copyfile(BASE_MODULAR_ENV_PATH, f"{os.path.join(ENV_DIR, env_name)}.py")

        params = {"xml": os.path.abspath(xml), "use_restricted_obs": use_restricted_obs}
        # register with gym
        register(
            id=f"{env_name}-v0",
            max_episode_steps=max_episode_steps,
            entry_point=f"environments.{env_name}:ModularEnv",
            kwargs=params,
        )
        env = wrappers.IdentityWrapper(gym.make(f"environments:{env_name}-v0"))
        # the following is the same for each env
        limb_obs_size = env.limb_obs_size
        max_action = env.max_action

    if limb_obs_size is None or max_action is None:
        raise RuntimeError("Failed to register any environments successfully.")

    return limb_obs_size, max_action


def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
    Args
    q: 1x4 quaternion
    Returns
    r: 1x3 exponential map
    Raises
    ValueError if the l2 norm of the quaternion is not close to 1
    """
    if np.abs(np.linalg.norm(q) - 1) > 1e-3:
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]
    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)
    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0
    r = r0 * theta
    return r


class ReplayBuffer(object):
    def __init__(self, buffer_size=1e6):
        self.size = int(buffer_size)
        self.count = 0
        self.real_size = 0

        self.obs_storage = None
        self.new_obs_storage = None
        self.action_storage = None
        self.reward_storage = None
        self.done_storage = None

    def initialize_storages(self, obs_shape, action_shape):
        self.obs_storage = np.zeros((self.size, *obs_shape), dtype=np.float32)
        self.new_obs_storage = np.zeros((self.size, *obs_shape), dtype=np.float32)
        self.action_storage = np.zeros((self.size, *action_shape), dtype=np.float32)
        self.reward_storage = np.zeros((self.size, 1), dtype=np.float32)
        self.done_storage = np.zeros((self.size, 1), dtype=np.float32)

    def add(self, data):
        if self.obs_storage is None:
            self.initialize_storages(data[0].shape, data[2].shape)
        obs, new_obs, action, reward, done = data
        self.obs_storage[self.count] = obs
        self.new_obs_storage[self.count] = new_obs
        self.action_storage[self.count] = action
        self.reward_storage[self.count] = reward
        self.done_storage[self.count] = done

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        sample_idxs = np.random.randint(0, self.real_size, size=batch_size)
        obs_batch = self.obs_storage[sample_idxs]
        new_obs_batch = self.new_obs_storage[sample_idxs]
        action_batch = self.action_storage[sample_idxs]
        reward_batch = self.reward_storage[sample_idxs]
        done_batch = self.done_storage[sample_idxs]
        batch = (
            obs_batch,
            new_obs_batch,
            action_batch,
            reward_batch,
            done_batch,
        )
        weights = np.ones_like(reward_batch)
        return batch, weights, None

    def update_priorities(self, tree_idxs, priorities):
        pass

    def __len__(self):
        return self.real_size


# credit: https://github.com/Howuhh/prioritized_experience_replay/blob/main/
class SumTree:
    def __init__(self, size):
        self.nodes = np.zeros(shape=(2 * size - 1), dtype=np.float32)
        self.data = np.array([None] * size)

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]
        self.nodes[idx] = value
        while idx != 0:
            idx = (idx - 1) // 2
            self.nodes[idx] += change

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total, f"cumsum: {cumsum}, total: {self.total}"
        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2 * idx + 1, 2 * idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum -= self.nodes[left]

        data_idx = idx - self.size + 1
        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size=1e6, eps=1e-2, alpha=0.1, beta=0.1):
        super().__init__(buffer_size)

        self.tree = SumTree(size=int(buffer_size))

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

    def add(self, data):
        self.tree.add(self.max_priority, self.count)
        super().add(data)

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "Buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = np.empty((batch_size, 1), dtype=np.float32)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = np.random.uniform(a, b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        batch = (
            self.obs_storage[sample_idxs],
            self.new_obs_storage[sample_idxs],
            self.action_storage[sample_idxs],
            self.reward_storage[sample_idxs],
            self.done_storage[sample_idxs]
        )
        return batch, weights, tree_idxs

    def update_priorities(self, tree_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for tree_idx, priority in zip(tree_idxs, priorities):
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(tree_idx, priority)
            self.max_priority = max(self.max_priority, priority)


class MLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MLPBase, self).__init__()
        self.l1 = nn.Linear(num_inputs, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, num_outputs)

    def forward(self, inputs):
        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


GLOBAL_SET_OF_NAMES = []


def getGraphStructure(xml_file, graph_type="morphology", return_action_ids=False):
    """Traverse the given xml file as a tree by pre-order and return the graph structure as a parents list"""

    # signal message flipping for flipped walker morphologies
    is_flipped = "walker" in os.path.basename(xml_file) and "flipped" in os.path.basename(
        xml_file
    )

    def preorder(b, parent_idx=-1):
        self_idx = len(parents)
        parents.append(parent_idx)
        b_name = b["@name"]
        if is_flipped:
            b_name += "_flipped"
        if b_name not in GLOBAL_SET_OF_NAMES:
            GLOBAL_SET_OF_NAMES.append(b_name)
        self_names.append(GLOBAL_SET_OF_NAMES.index(b_name))
        if "body" not in b:
            return
        if not isinstance(b["body"], list):
            b["body"] = [b["body"]]
        for branch in b["body"]:
            preorder(branch, self_idx)

    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    parents = []
    self_names = []
    try:
        root = xml["mujoco"]["worldbody"]["body"]
        assert not isinstance(
            root, list
        ), "worldbody can only contain one body (torso) for the current implementation, but found {}".format(
            root
        )
    except:
        raise Exception(
            "The given xml file does not follow the standard MuJoCo format."
        )
    preorder(root)
    # signal message flipping for flipped walker morphologies
    if is_flipped:
        parents[0] = -2

    if graph_type == "tree":
        parents[1:] = [0] * len(parents[1:])
    elif graph_type == "line":
        for i in range(1, len(parents)):
            parents[i] = i - 1

    if return_action_ids:
        return parents, self_names
    else:
        return parents


def weighted_mse_loss(input, target, weight):
    error = input - target
    return torch.mean(weight * error ** 2), error.detach().cpu()


def getGraphJoints(xml_file):
    """Traverse the given xml file as a tree by pre-order and return all the joints defined as a list of tuples (body_name, joint_name1, ...) for each body"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""

    def preorder(b):
        if "joint" in b:
            if isinstance(b["joint"], list) and b["@name"] != "torso":
                raise Exception(
                    "The given xml file does not follow the standard MuJoCo format."
                )
            elif not isinstance(b["joint"], list):
                b["joint"] = [b["joint"]]
            joints.append([b["@name"]])
            for j in b["joint"]:
                joints[-1].append(j["@name"])
        if "body" not in b:
            return
        if not isinstance(b["body"], list):
            b["body"] = [b["body"]]
        for branch in b["body"]:
            preorder(branch)

    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    try:
        root = xml["mujoco"]["worldbody"]["body"]
    except:
        raise Exception(
            "The given xml file does not follow the standard MuJoCo format."
        )
    preorder(root)
    return joints


def getMotorJoints(xml_file):
    """Traverse the given xml file as a tree by pre-order and return the joint names in the order of defined actuators"""
    """Used to match the order of joints defined in worldbody and joints defined in actuators"""
    with open(xml_file) as fd:
        xml = xmltodict.parse(fd.read())
    joints = []
    motors = xml["mujoco"]["actuator"]["motor"]
    if not isinstance(motors, list):
        motors = [motors]
    for m in motors:
        joints.append(m["@joint"])
    return joints
