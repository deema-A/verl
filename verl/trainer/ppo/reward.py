# Copyright 2025 Individual Contributor: Thibaut Barroyer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import ray

from verl import DataProto


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    print(f"Deema Using reward manager: {reward_manager_name}")
    if reward_manager_name == "naive":
        from verl.workers.reward_manager import NaiveRewardManager

        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    elif reward_manager_name == "batch":
        from verl.workers.reward_manager import BatchRewardManager

        reward_manager_cls = BatchRewardManager
    elif reward_manager_name == "dapo":
        from verl.workers.reward_manager import DAPORewardManager

        reward_manager_cls = DAPORewardManager
    elif reward_manager_name == "trek":
        from verl.workers.reward_manager import TREKRewardManager

        reward_manager_cls = TREKRewardManager
    else:
        raise NotImplementedError

    compute_score = get_custom_reward_fn(config)
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )


def compute_reward(data: DataProto, reward_fn):
    """
    Compute reward for a batch of data.
    Args:
        data: DataProto object containing the input data.
        reward_fn: Reward function to compute the reward.
    Returns:
        Tuple of reward tensor and extra info dictionary.
    """
    try:
        print("Deema I am here") # was here
        # print(f"Computing reward for data: {data.batch}")
        """
        Computing reward for data: TensorDict(
        fields={
            attention_mask: Tensor(shape=torch.Size([256, 768]), device=cpu, dtype=torch.int64, is_shared=False),
            input_ids: Tensor(shape=torch.Size([256, 768]), device=cpu, dtype=torch.int64, is_shared=False),
            position_ids: Tensor(shape=torch.Size([256, 768]), device=cpu, dtype=torch.int64, is_shared=False),
            prompts: Tensor(shape=torch.Size([256, 512]), device=cpu, dtype=torch.int64, is_shared=False),
            response_mask: Tensor(shape=torch.Size([256, 256]), device=cpu, dtype=torch.int64, is_shared=False),
            responses: Tensor(shape=torch.Size([256, 256]), device=cpu, dtype=torch.int64, is_shared=False)
        },
        batch_size=torch.Size([256]),
        device=None,
        is_shared=False)
        """
        # print(f"Data batch keys: {data.batch.keys()}")
        """
        _StringKeys(dict_keys(['responses', 'attention_mask', 'input_ids', 'position_ids', 'prompts', 'response_mask']))
        """
        # print(f"data.batch['responses'][2]: {data.batch['responses'][2]}")
        # tensor([  1249,   8253,   1246,   1657,  18636,
        # print(f"data.batch['prompts'][2]: {data.batch['prompts'][2]}")
        # tensor([151643, 151643, 151643, 151643, 151643, 1516
        print(f"reward_fn: {reward_fn}")
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result["reward_extra_info"]
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}

    return reward_tensor, reward_extra_infos_dict


@ray.remote(num_cpus=1)
def compute_reward_async(data: DataProto, config, tokenizer):
    """
    Load the reward manager and compute the reward for a batch of data.
    This is meant to be run in a separate Ray worker.
    """
    reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
    return compute_reward(data, reward_fn)
