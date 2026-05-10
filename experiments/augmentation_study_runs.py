import inspect

from sen2classification.augmentations import augment_boa_and_time


AUGMENTATION_LABELS = {
    "p_random_noise": "random noise",
    "p_constant_offset": "constant offset",
    "p_time_jitter": "time jitter",
    "p_time_dependent_noise": "time-dependent noise",
    "p_blackout": "blackout",
    "p_gamma": "gamma correction",
    "p_cloud_simulation": "cloud simulation",
    "p_cloud_shadow": "cloud shadow",
    "p_observation_dropout": "observation dropout",
    "p_vegetation_period_modify": "vegetation period shift",
}


AUGMENTATION_SWEEP_VALUES = {
    # Already tested.
    # "p_random_noise": ("noise_scale", (0.01, 0.02, 0.04)),
    # Already tested.
    # "p_constant_offset": ("offset_scale", (0.01, 0.02, 0.04)),
    # Best current setting: time_jitter_max=14.
    # "p_time_jitter": ("time_jitter_max", (7, 14, 28)),
    "p_time_jitter": ("time_jitter_max", (10, 18, 42)),
    # Already tested.
    # "p_time_dependent_noise": ("time_noise_strength", (0.01, 0.02, 0.04)),
    # Best current setting: blackout_percentage=0.05.
    # "p_blackout": ("blackout_percentage", (0.01, 0.02, 0.05)),
    "p_blackout": ("blackout_percentage", (0.03, 0.07, 0.15)),
    # Already tested.
    # "p_gamma": ("gamma_offset", (0.001, 0.002, 0.005)),
    # Best current setting: dropout_percentage=0.3.
    # "p_observation_dropout": ("dropout_percentage", (0.1, 0.2, 0.3)),
    "p_observation_dropout": ("dropout_percentage", (0.25, 0.35, 0.5)),
    # Best current setting: veg_period_max_delta=10.
    # "p_vegetation_period_modify": ("veg_period_max_delta", (5, 10, 20)),
    "p_vegetation_period_modify": ("veg_period_max_delta", (8, 14, 30)),
}


CLOUD_PROBABILITY_TESTS = {
    # Already tested.
    # "p_cloud_simulation": 0.02,
    # Already tested.
    # "p_cloud_shadow": 0.02,
}


def format_value_for_name(value):
    return str(value).replace(".", "p")


def build_run(name, label, group, augmentation_kwargs, variant=None, setting=None):
    return {
        "name": name,
        "label": label,
        "group": group,
        "variant": label if variant is None else variant,
        "setting": "" if setting is None else setting,
        "augmentation_kwargs": augmentation_kwargs,
    }


def build_single_augmentation_runs(disabled):
    runs = []
    for prob_key, (value_key, values) in AUGMENTATION_SWEEP_VALUES.items():
        augmentation_name = prob_key.removeprefix("p_")
        augmentation_label = AUGMENTATION_LABELS[prob_key]
        for value in values:
            runs.append(
                build_run(
                    name=f"only_{augmentation_name}_{value_key}_{format_value_for_name(value)}",
                    label=f"{augmentation_label} ({value_key}={value})",
                    group="single",
                    variant=augmentation_label,
                    setting=f"{value_key}={value}",
                    augmentation_kwargs=disabled | {prob_key: 1.0, value_key: value},
                )
            )
    return runs


def build_probability_only_runs(disabled):
    runs = []
    for prob_key, probability in CLOUD_PROBABILITY_TESTS.items():
        augmentation_name = prob_key.removeprefix("p_")
        augmentation_label = AUGMENTATION_LABELS[prob_key]
        runs.append(
            build_run(
                name=f"only_{augmentation_name}_{prob_key}_{format_value_for_name(probability)}",
                label=f"{augmentation_label} ({prob_key}={probability})",
                group="single",
                variant=augmentation_label,
                setting=f"{prob_key}={probability}",
                augmentation_kwargs=disabled | {prob_key: probability},
            )
        )
    return runs


def get_default_augmentation_kwargs():
    defaults = {}
    for name, parameter in inspect.signature(augment_boa_and_time).parameters.items():
        if parameter.default is inspect.Signature.empty or name == "rng":
            continue
        defaults[name] = parameter.default
    return defaults


def get_probability_keys():
    defaults = get_default_augmentation_kwargs()
    return tuple(name for name in defaults if name.startswith("p_"))


def get_disabled_augmentation_kwargs():
    defaults = get_default_augmentation_kwargs()
    return defaults | {name: 0.0 for name in get_probability_keys()}


def build_runs():
    disabled = get_disabled_augmentation_kwargs()

    runs = []

    # Already tested.
    # runs.append(
    #     build_run(
    #         name="baseline_no_aug",
    #         label="baseline",
    #         group="baseline",
    #         variant="baseline",
    #         setting="none",
    #         augmentation_kwargs=disabled,
    #     )
    # )

    # Already tested.
    # defaults = get_default_augmentation_kwargs()
    # runs.extend(
    #     [
    #         build_run(
    #             name="all_augmentations_default_values",
    #             label="all augmentations (defaults)",
    #             group="final",
    #             variant="all augmentations",
    #             setting="defaults",
    #             augmentation_kwargs=defaults,
    #         ),
    #     ]
    # )

    runs.extend(build_single_augmentation_runs(disabled))

    # Already tested.
    # runs.extend(build_probability_only_runs(disabled))

    return runs