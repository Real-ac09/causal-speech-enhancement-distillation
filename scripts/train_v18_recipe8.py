#!/usr/bin/env python3
"""Train Recipe 8 with balanced prefix-policy supervision."""

from __future__ import annotations

import train as base
from train_v17_recipe5b import make_loader


if __name__ == "__main__":
    base.make_loader = make_loader
    base.main()
