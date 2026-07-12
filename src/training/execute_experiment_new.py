from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

# Add parent directories to path for direct script execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "src" / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "training"))

try:
    from .experiment_config import build_arg_parser, build_config_from_args
    from .training_engine import ExperimentRunner
except ImportError:
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
