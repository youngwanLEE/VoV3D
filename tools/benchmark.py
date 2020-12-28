#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
A script to benchmark data loading.
"""

import vov3d.utils.logging as logging
from vov3d.utils.benchmark import benchmark_data_loading
from vov3d.utils.misc import launch_job
from vov3d.utils.parser import load_config, parse_args

logger = logging.get_logger(__name__)


def main():
    args = parse_args()
    cfg = load_config(args)

    launch_job(
        cfg=cfg, init_method=args.init_method, func=benchmark_data_loading
    )


if __name__ == "__main__":
    main()
