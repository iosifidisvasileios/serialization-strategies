from __future__ import annotations
from typing import Optional
from experiment_config import build_arg_parser, build_config_from_args
from training_engine import ExperimentRunner


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = build_config_from_args(args)
    runner = ExperimentRunner(config)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
