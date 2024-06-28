from __future__ import print_function
import os
import torch
import utils
import numpy as np


def has_checkpoint(checkpoint_path, rb_path):
    """check if a checkpoint exists"""
    if not (os.path.exists(checkpoint_path) and os.path.exists(rb_path)):
        return False
    if "model.pyth" not in os.listdir(checkpoint_path):
        return False
    if len(os.listdir(rb_path)) == 0:
        return False
    return True


def save_model(
    checkpoint_path,
    policy,
    total_timesteps,
    episode_num,
    num_samples,
    replay_buffer,
    env_names,
    args,
    model_name="model.pyth",
):
    # change to default graph before saving
    policy.change_morphology([-1], [0])
    # Record the state
    checkpoint = {
        "actor_state": policy.actor.state_dict(),
        "critic_state": policy.critic.state_dict(),
        "actor_target_state": policy.actor_target.state_dict(),
        "critic_target_state": policy.critic_target.state_dict(),
        "actor_optimizer_state": policy.actor_optimizer.state_dict(),
        "critic_optimizer_state": policy.critic_optimizer.state_dict(),
        "total_timesteps": total_timesteps,
        "episode_num": episode_num,
        "num_samples": num_samples,
        "args": args,
        "rb_max": {name: replay_buffer[name].max_size for name in replay_buffer},
        "rb_ptr": {name: replay_buffer[name].ptr for name in replay_buffer},
        "rb_real": {name: replay_buffer[name].real_size for name in replay_buffer}
    }
    fpath = os.path.join(checkpoint_path, model_name)
    # (over)write the checkpoint
    torch.save(checkpoint, fpath)
    return fpath


def save_replay_buffer(rb_path, replay_buffer):
    # save replay buffer
    for name in replay_buffer:
        path = os.path.join(rb_path, f'{name}')
        np.save(path+'-obs.npy', replay_buffer[name].obs_storage)
        np.save(path+'-new_obs.npy', replay_buffer[name].new_obs_storage)
        np.save(path+'-action.npy', replay_buffer[name].action_storage)
        np.save(path+'-reward.npy', replay_buffer[name].reward_storage)
        np.save(path+'-done.npy', replay_buffer[name].done_storage)
    return rb_path


def load_checkpoint(checkpoint_path, rb_path, policy, args):
    fpath = os.path.join(checkpoint_path, "model.pyth")
    checkpoint = torch.load(fpath, map_location="cpu")
    # change to default graph before loading
    policy.change_morphology([-1], [0])
    # load and return checkpoint
    policy.actor.load_state_dict(checkpoint["actor_state"])
    policy.critic.load_state_dict(checkpoint["critic_state"])
    policy.actor_target.load_state_dict(checkpoint["actor_target_state"])
    policy.critic_target.load_state_dict(checkpoint["critic_target_state"])
    policy.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state"])
    policy.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state"])
    # load replay buffer
    all_rb_files = [f[:-4] for f in os.listdir(rb_path) if ".npy" in f]
    all_rb_files.sort()
    replay_buffer_new: {str: utils.ReplayBuffer} = dict()
    for name in all_rb_files:
        if len(all_rb_files) > args.rb_max // 1e6:
            replay_buffer_new[name] = utils.ReplayBuffer(buffer_size=args.rb_max // len(all_rb_files))
        else:
            replay_buffer_new[name] = utils.ReplayBuffer()
        replay_buffer_new[name].size = int(checkpoint["rb_max"][name])
        replay_buffer_new[name].count = int(checkpoint["rb_ptr"][name])
        replay_buffer_new[name].count = int(checkpoint["rb_real"][name])
        replay_buffer_new[name].obs_storage = np.load(rb_path+'-obs.npy')
        replay_buffer_new[name].new_obs_storage = np.load(rb_path+'-new_obs.npy')
        replay_buffer_new[name].action_storage = np.load(rb_path+'-action.npy')
        replay_buffer_new[name].reward_storage = np.load(rb_path+'-reward.npy')
        replay_buffer_new[name].done_storage = np.load(rb_path+'-done.npy')


    return (
            checkpoint["total_timesteps"],
            checkpoint["episode_num"],
            replay_buffer_new,
            checkpoint["num_samples"],
            fpath,
        )


def load_model_only(exp_path, policy, model_name="model.pyth"):
    model_path = os.path.join(exp_path, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError("no model file found")
    print("*** using model {} ***".format(model_path))
    checkpoint = torch.load(model_path, map_location="cpu")
    # change to default graph before loading
    policy.change_morphology([-1], [0])
    # load and return checkpoint
    policy.actor.load_state_dict(checkpoint["actor_state"])
    policy.critic.load_state_dict(checkpoint["critic_state"])
    policy.actor_target.load_state_dict(checkpoint["actor_target_state"])
    policy.critic_target.load_state_dict(checkpoint["critic_target_state"])
    policy.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state"])
    policy.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state"])