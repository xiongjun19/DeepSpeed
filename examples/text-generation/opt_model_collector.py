# coding=utf8

import os
import torch
from torch import nn
from typing import Optional
from typing import Union
from transformers.configuration_utils import PretrainedConfig
from transformers import OPTForCausalLM 


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False


def get_config(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
    config = kwargs.pop("config", None)
    state_dict = kwargs.pop("state_dict", None)
    cache_dir = kwargs.pop("cache_dir", None)
    from_tf = kwargs.pop("from_tf", False)
    from_flax = kwargs.pop("from_flax", False)
    ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    output_loading_info = kwargs.pop("output_loading_info", False)
    local_files_only = kwargs.pop("local_files_only", False)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    mirror = kwargs.pop("mirror", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    _fast_init = kwargs.pop("_fast_init", True)
    torch_dtype = kwargs.pop("torch_dtype", None)
    low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)

    from_pt = not (from_tf | from_flax)

    user_agent = {
            "file_type": "model",
            "framework": "pytorch",
            "from_auto_class": from_auto_class,
    }
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    if _is_offline_mode and not local_files_only:
        local_files_only = True

    # Load config if we don't provide a configuration
    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else pretrained_model_name_or_path
        config, model_kwargs = cls.config_class.from_pretrained(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
            **kwargs,
        )
    else:
        model_kwargs = kwargs
    return config, model_args, model_kwargs


def test(model_name, output_path):
    model_cfg, model_args, model_kwargs = get_config(
            OPTForCausalLM,
            model_name)
    org_model = OPTForCausalLM(model_cfg, *model_args, **model_kwargs).half()
    torch.save(org_model, output_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, help="path to save the model")
    parser.add_argument('-m', '--model_name', type=str, help="the model name of gpt")
    args = parser.parse_args()
    test(args.model_name, args.output)
