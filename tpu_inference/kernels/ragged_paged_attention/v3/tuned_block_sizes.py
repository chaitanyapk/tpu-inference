# Copyright 2025 Google LLC
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
"""Auto-tuned block sizes for ragged paged attention."""

import jax.numpy as jnp

from tpu_inference.kernels.ragged_paged_attention.v3.util import (
    align_to, get_dtype_packing, get_tpu_version, next_power_of_2)
from tpu_inference.logger import init_logger
from tpu_inference.utils import get_device_name

logger = init_logger(__name__)

# key
#   - device_name
#     - page_size
#       - q_{q_dtype_name}_kv_{kv_dtype_name}
#         - q_head-{num_q_heads}_kv_head-{num_kv_heads}-_head-{head_dim}
#           - max_model_len-{max_model_len}-sw-{sliding_window}
# value:
#   - (num_kv_pages_per_block, num_queries_per_block)
TUNED_BLOCK_SIZES = {
    'TPU v7': {
  "128": {
    "q_bfloat16_kv_bfloat16": {
      "q_head-128_kv_head-16_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-16_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-2_head-64": {
        "max_model_len-1024-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 64
          }
        }
      },
      "q_head-16_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 64
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 64
          }
        }
      },
      "q_head-16_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        }
      },
      "q_head-2_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 128
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 256
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 128
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-2_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-32_kv_head-16_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-16_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-32_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-4_head-64": {
        "max_model_len-1024-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-4_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 128
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-4_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-4_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 128
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-4_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-64_kv_head-16_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-16_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-32_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-64_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-8_head-64": {
        "max_model_len-1024-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-8_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-2_head-64": {
        "max_model_len-1024-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-128": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 128
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 128
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-8_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        }
      }
    },
    "q_bfloat16_kv_float8_e4m3fn": {
      "q_head-128_kv_head-16_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-16_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 64
          }
        }
      },
      "q_head-16_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-2_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 128
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        }
      },
      "q_head-2_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-32_kv_head-16_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-32_kv_head-16_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-32_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-4_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 128
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 128
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-4_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-64_kv_head-16_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-16_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-32_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-64_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-8_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-8_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-128-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 32,
            "num_q_per_block": 64
          }
        }
      }
    }
  },
  "16": {
    "q_bfloat16_kv_bfloat16": {
      "q_head-8_kv_head-2_head-128": {
        "max_model_len-16384-sw-None": {
          "config": {
            "num_kv_pages_per_block": 128,
            "num_q_per_block": 256
          },
          "metadata": {
            "benchmarking_method": "amortized",
            "num_repeats": 5,
            "samples_ns": [
              49580.32,
              48735.13,
              48541.85,
              48566.09,
              48490.24
            ]
          },
          "stats": {
            "compile_time_s": 3.996254135010531,
            "latency_avg_ns": 48782.725999999995,
            "latency_std_ns": 455.2257769832469,
            "lower_time_s": 1.1132908850267995
          }
        }
      }
    },
    "q_bfloat16_kv_float8_e4m3fn": {
      "q_head-8_kv_head-2_head-128": {
        "max_model_len-16384-sw-None": {
          "config": {
            "num_kv_pages_per_block": 128,
            "num_q_per_block": 256
          },
          "metadata": {
            "benchmarking_method": "amortized",
            "num_repeats": 5,
            "samples_ns": [
              47639.12,
              47237.96,
              47272.48,
              47128.33,
              46967.09
            ]
          },
          "stats": {
            "compile_time_s": 3.849032800993882,
            "latency_avg_ns": 47248.996,
            "latency_std_ns": 248.44582071349265,
            "lower_time_s": 1.1189253270276822
          }
        }
      }
    }
  },
  "256": {
    "q_bfloat16_kv_bfloat16": {
      "q_head-128_kv_head-16_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-16_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-2_head-64": {
        "max_model_len-1024-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        }
      },
      "q_head-16_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        }
      },
      "q_head-16_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        }
      },
      "q_head-2_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 128
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-2_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-32_kv_head-16_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-16_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-32_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-32_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-4_head-64": {
        "max_model_len-1024-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-4_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 128
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-4_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-4_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-4_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-64_kv_head-16_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-16_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-32_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-64_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-8_head-64": {
        "max_model_len-1024-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-8_kv_head-1_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-1_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-2_head-64": {
        "max_model_len-1024-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-128": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 128
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-8_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        }
      }
    },
    "q_bfloat16_kv_float8_e4m3fn": {
      "q_head-128_kv_head-16_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-128_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-128_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-16_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-16_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        }
      },
      "q_head-16_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-2_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 128
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        }
      },
      "q_head-2_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 64
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-16_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-32_kv_head-16_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-32_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-32_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-4_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 128
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 128
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-4_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        }
      },
      "q_head-64_kv_head-16_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-16_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-32_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        }
      },
      "q_head-64_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-64_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 8
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 8
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 8
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 8
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-8_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-64_kv_head-8_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 16
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 16
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 16
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 16
          }
        }
      },
      "q_head-8_kv_head-2_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-16384-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 256
          },
          "metadata": {
            "benchmarking_method": "amortized",
            "num_repeats": 5,
            "samples_ns": [
              41086.44,
              40656.51,
              40476.99,
              40642.71,
              40569.06
            ]
          },
          "stats": {
            "compile_time_s": 3.6265507079660892,
            "latency_avg_ns": 40686.342,
            "latency_std_ns": 234.7634110546207,
            "lower_time_s": 0.3320452229818329
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 128
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-2_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 32
          }
        }
      },
      "q_head-8_kv_head-4_head-128": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 64
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 64
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 64
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 128
          }
        }
      },
      "q_head-8_kv_head-4_head-256": {
        "max_model_len-1024-sw-None": {
          "config": {
            "num_kv_pages_per_block": 4,
            "num_q_per_block": 32
          }
        },
        "max_model_len-2048-sw-None": {
          "config": {
            "num_kv_pages_per_block": 8,
            "num_q_per_block": 32
          }
        },
        "max_model_len-256-sw-None": {
          "config": {
            "num_kv_pages_per_block": 1,
            "num_q_per_block": 32
          }
        },
        "max_model_len-4096-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        },
        "max_model_len-512-sw-None": {
          "config": {
            "num_kv_pages_per_block": 2,
            "num_q_per_block": 32
          }
        },
        "max_model_len-8192-sw-None": {
          "config": {
            "num_kv_pages_per_block": 16,
            "num_q_per_block": 64
          }
        }
      }
    }
  }
    },
    'TPU v6e': {
        128: {
            'q_bfloat16_kv_float8_e4m3fn': {
                'q_head-8_kv_head-4_head-256': {
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-8192-sw-None': (32, 128),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 64),
                },
                'q_head-16_kv_head-4_head-128': {
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                    'max_model_len-128-sw-None': (1, 32),
                },
                'q_head-32_kv_head-16_head-256': {
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                },
                'q_head-32_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                },
                'q_head-64_kv_head-2_head-128': {
                    'max_model_len-4096-sw-None': (32, 16),
                    'max_model_len-8192-sw-None': (32, 16),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (16, 16),
                },
                'q_head-64_kv_head-16_head-128': {
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (1, 16),
                },
                'q_head-128_kv_head-8_head-256': {
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-4096-sw-None': (16, 8),
                    'max_model_len-8192-sw-None': (16, 8),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                },
                'q_head-4_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-2048-sw-None': (16, 64),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-8192-sw-None': (32, 128),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (4, 32),
                },
                'q_head-128_kv_head-2_head-128': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-64_kv_head-2_head-256': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (32, 8),
                    'max_model_len-8192-sw-None': (32, 16),
                },
                'q_head-128_kv_head-16_head-128': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-4096-sw-None': (16, 8),
                    'max_model_len-8192-sw-None': (16, 8),
                },
                'q_head-32_kv_head-4_head-128': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-8_kv_head-4_head-128': {
                    'max_model_len-128-sw-None': (1, 64),
                    'max_model_len-256-sw-None': (2, 64),
                    'max_model_len-512-sw-None': (4, 64),
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-2048-sw-None': (16, 64),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-8192-sw-None': (32, 128),
                },
                'q_head-8_kv_head-2_head-128': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-64_kv_head-16_head-256': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-4096-sw-None': (8, 8),
                    'max_model_len-8192-sw-None': (8, 8),
                },
                'q_head-2_kv_head-2_head-128': {
                    'max_model_len-128-sw-None': (1, 64),
                    'max_model_len-256-sw-None': (2, 128),
                    'max_model_len-512-sw-None': (4, 64),
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-2048-sw-None': (16, 128),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-8192-sw-None': (32, 128),
                },
                'q_head-16_kv_head-4_head-256': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-64_kv_head-32_head-128': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-4096-sw-None': (8, 8),
                    'max_model_len-8192-sw-None': (8, 8),
                },
                'q_head-16_kv_head-2_head-256': {
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-32_kv_head-16_head-128': {
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                },
                'q_head-32_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-64_kv_head-8_head-256': {
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-128_kv_head-8_head-128': {
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (32, 8),
                    'max_model_len-8192-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                },
                'q_head-128_kv_head-2_head-256': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-4096-sw-None': (32, 8),
                    'max_model_len-8192-sw-None': (32, 8),
                },
                'q_head-16_kv_head-8_head-128': {
                    'max_model_len-128-sw-None': (1, 64),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 64),
                    'max_model_len-4096-sw-None': (32, 64),
                    'max_model_len-8192-sw-None': (32, 64),
                },
                'q_head-64_kv_head-4_head-128': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (32, 16),
                    'max_model_len-8192-sw-None': (32, 16),
                },
                'q_head-16_kv_head-2_head-128': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-32_kv_head-4_head-256': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-8_kv_head-2_head-256': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-8192-sw-None': (32, 128),
                },
                'q_head-2_kv_head-2_head-256': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 64),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-8192-sw-None': (32, 128),
                },
                'q_head-128_kv_head-4_head-128': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (32, 8),
                    'max_model_len-8192-sw-None': (32, 8),
                },
                'q_head-4_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-128-sw-None': (1, 64),
                    'max_model_len-256-sw-None': (2, 128),
                    'max_model_len-512-sw-None': (4, 128),
                    'max_model_len-2048-sw-None': (16, 128),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-8192-sw-None': (32, 128),
                },
                'q_head-16_kv_head-8_head-256': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 64),
                    'max_model_len-4096-sw-None': (16, 64),
                    'max_model_len-8192-sw-None': (16, 64),
                },
                'q_head-64_kv_head-4_head-256': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-32_kv_head-8_head-128': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-128_kv_head-4_head-256': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-64_kv_head-8_head-128': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 16),
                    'max_model_len-8192-sw-None': (32, 16),
                },
                'q_head-32_kv_head-8_head-256': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
            },
            'q_bfloat16_kv_bfloat16': {
                'q_head-8_kv_head-2_head-128': {
                    'max_model_len-8192-sw-None': (32, 128),
                    'max_model_len-128-sw-None': (1, 64),
                    'max_model_len-512-sw-None': (4, 64),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 64),
                },
                'q_head-16_kv_head-2_head-128': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-16_kv_head-8_head-256': {
                    'max_model_len-8192-sw-None': (8, 64),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-4096-sw-None': (8, 64),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 64),
                },
                'q_head-32_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                },
                'q_head-32_kv_head-8_head-256': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-64_kv_head-1_head-128': {
                    'max_model_len-4096-sw-None': (32, 16),
                    'max_model_len-8192-sw-None': (32, 32),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (16, 16),
                },
                'q_head-64_kv_head-8_head-128': {
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 16),
                },
                'q_head-128_kv_head-4_head-256': {
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 16),
                },
                'q_head-128_kv_head-1_head-128': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-4096-sw-None': (32, 16),
                    'max_model_len-8192-sw-None': (32, 16),
                },
                'q_head-8_kv_head-2_head-256': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 128),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-32_kv_head-16_head-128': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-2_kv_head-1_head-256': {
                    'max_model_len-128-sw-None': (1, 64),
                    'max_model_len-512-sw-None': (4, 128),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-256-sw-None': (2, 64),
                    'max_model_len-1024-sw-None': (8, 128),
                    'max_model_len-8192-sw-None': (32, 128),
                    'max_model_len-2048-sw-None': (16, 128),
                },
                'q_head-2_kv_head-1_head-128': {
                    'max_model_len-512-sw-None': (4, 128),
                    'max_model_len-256-sw-None': (2, 128),
                    'max_model_len-2048-sw-None': (16, 128),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-128-sw-None': (1, 64),
                    'max_model_len-8192-sw-None': (32, 128),
                    'max_model_len-1024-sw-None': (8, 128),
                },
                'q_head-8_kv_head-1_head-256': {
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-64_kv_head-1_head-256': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (32, 16),
                    'max_model_len-8192-sw-None': (32, 16),
                },
                'q_head-4_kv_head-2_head-128': {
                    'max_model_len-8192-sw-None': (32, 128),
                    'max_model_len-128-sw-None': (1, 128),
                    'max_model_len-512-sw-None': (4, 128),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-256-sw-None': (2, 64),
                    'max_model_len-1024-sw-None': (8, 128),
                    'max_model_len-2048-sw-None': (16, 128),
                },
                'q_head-4_kv_head-2_head-256': {
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (8, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                    'max_model_len-512-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (16, 128),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (16, 128),
                },
                'q_head-8_kv_head-4_head-128': {
                    'max_model_len-512-sw-None': (4, 128),
                    'max_model_len-128-sw-None': (1, 64),
                    'max_model_len-256-sw-None': (2, 64),
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-2048-sw-None': (16, 128),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (32, 128),
                },
                'q_head-8_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (8, 128),
                    'max_model_len-2048-sw-None': (16, 128),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-8_kv_head-1_head-128': {
                    'max_model_len-256-sw-None': (2, 64),
                    'max_model_len-8192-sw-None': (32, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                },
                'q_head-32_kv_head-2_head-128': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-128_kv_head-8_head-128': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-64_kv_head-8_head-256': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-16_kv_head-2_head-256': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-4_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (8, 128),
                    'max_model_len-8192-sw-None': (32, 128),
                    'max_model_len-2048-sw-None': (16, 128),
                    'max_model_len-128-sw-None': (1, 128),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-256-sw-None': (2, 128),
                    'max_model_len-512-sw-None': (4, 128),
                },
                'q_head-16_kv_head-1_head-256': {
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-16_kv_head-8_head-128': {
                    'max_model_len-8192-sw-None': (16, 128),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 64),
                    'max_model_len-512-sw-None': (4, 64),
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-2048-sw-None': (16, 128),
                    'max_model_len-4096-sw-None': (16, 128),
                },
                'q_head-32_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                    'max_model_len-512-sw-None': (4, 32),
                },
                'q_head-32_kv_head-8_head-128': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 64),
                    'max_model_len-4096-sw-None': (16, 64),
                    'max_model_len-8192-sw-None': (16, 64),
                },
                'q_head-64_kv_head-4_head-256': {
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-128_kv_head-4_head-128': {
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                },
                'q_head-4_kv_head-1_head-256': {
                    'max_model_len-2048-sw-None': (16, 64),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (32, 128),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                },
                'q_head-128_kv_head-1_head-256': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-4096-sw-None': (32, 8),
                    'max_model_len-8192-sw-None': (32, 8),
                },
                'q_head-32_kv_head-16_head-256': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (4, 16),
                },
                'q_head-64_kv_head-2_head-128': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 16),
                    'max_model_len-8192-sw-None': (32, 16),
                },
                'q_head-128_kv_head-8_head-256': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-4096-sw-None': (8, 8),
                    'max_model_len-8192-sw-None': (8, 8),
                },
                'q_head-32_kv_head-2_head-256': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-16_head-128': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-16_kv_head-4_head-128': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 64),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 128),
                    'max_model_len-4096-sw-None': (16, 64),
                    'max_model_len-8192-sw-None': (16, 64),
                },
                'q_head-128_kv_head-2_head-128': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (32, 8),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-64_kv_head-2_head-256': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-128_kv_head-16_head-128': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-4096-sw-None': (8, 8),
                    'max_model_len-8192-sw-None': (8, 8),
                },
                'q_head-32_kv_head-4_head-128': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-16_head-256': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (4, 8),
                },
                'q_head-16_kv_head-4_head-256': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-2048-sw-None': (16, 128),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-64_kv_head-32_head-128': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (4, 8),
                },
                'q_head-128_kv_head-2_head-256': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-4096-sw-None': (32, 8),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-64_kv_head-4_head-128': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-16_kv_head-1_head-128': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-32_kv_head-4_head-256': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
            },
        },
        256: {
            'q_bfloat16_kv_bfloat16': {
                'q_head-2_kv_head-1_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-1024-sw-None': (4, 128),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-2_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (4, 128),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-4096-sw-None': (16, 64),
                    'max_model_len-8192-sw-None': (16, 64),
                    'max_model_len-256-sw-None': (1, 128),
                    'max_model_len-512-sw-None': (2, 128),
                },
                'q_head-8_kv_head-4_head-256': {
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-4096-sw-None': (8, 128),
                    'max_model_len-8192-sw-None': (4, 128),
                    'max_model_len-256-sw-None': (1, 32),
                },
                'q_head-8_kv_head-4_head-128': {
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-4096-sw-None': (8, 128),
                    'max_model_len-8192-sw-None': (8, 128),
                    'max_model_len-512-sw-None': (2, 128),
                    'max_model_len-1024-sw-None': (4, 128),
                    'max_model_len-256-sw-None': (1, 64),
                },
                'q_head-16_kv_head-2_head-256': {
                    'max_model_len-8192-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                },
                'q_head-32_kv_head-16_head-256': {
                    'max_model_len-2048-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (2, 16),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (2, 16),
                },
                'q_head-32_kv_head-16_head-128': {
                    'max_model_len-4096-sw-None': (4, 32),
                    'max_model_len-8192-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                },
                'q_head-64_kv_head-2_head-128': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-128_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-4096-sw-None': (16, 8),
                    'max_model_len-8192-sw-None': (16, 8),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                },
                'q_head-128_kv_head-8_head-256': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (4, 8),
                },
                'q_head-32_kv_head-2_head-256': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-64_kv_head-16_head-128': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (4, 16),
                },
                'q_head-16_kv_head-4_head-128': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-4096-sw-None': (8, 64),
                    'max_model_len-8192-sw-None': (8, 128),
                },
                'q_head-4_kv_head-1_head-128': {
                    'max_model_len-256-sw-None': (1, 128),
                    'max_model_len-512-sw-None': (2, 128),
                    'max_model_len-1024-sw-None': (4, 128),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-128_kv_head-2_head-128': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (16, 8),
                    'max_model_len-8192-sw-None': (16, 8),
                },
                'q_head-64_kv_head-2_head-256': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-128_kv_head-16_head-128': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (4, 8),
                },
                'q_head-8_kv_head-2_head-256': {
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-4096-sw-None': (8, 128),
                    'max_model_len-8192-sw-None': (8, 128),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                },
                'q_head-16_kv_head-2_head-128': {
                    'max_model_len-8192-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                },
                'q_head-32_kv_head-8_head-256': {
                    'max_model_len-4096-sw-None': (4, 32),
                    'max_model_len-8192-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (4, 32),
                },
                'q_head-64_kv_head-1_head-256': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-128_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (16, 16),
                    'max_model_len-512-sw-None': (2, 8),
                },
                'q_head-128_kv_head-8_head-128': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-8_kv_head-1_head-128': {
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-32_kv_head-4_head-128': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-64_kv_head-16_head-256': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (2, 8),
                    'max_model_len-2048-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (2, 8),
                    'max_model_len-8192-sw-None': (2, 8),
                },
                'q_head-16_kv_head-4_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-4096-sw-None': (8, 128),
                    'max_model_len-8192-sw-None': (8, 128),
                },
                'q_head-4_kv_head-1_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-64_kv_head-32_head-128': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (2, 8),
                    'max_model_len-2048-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (2, 8),
                    'max_model_len-8192-sw-None': (2, 8),
                },
                'q_head-16_kv_head-8_head-128': {
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-1024-sw-None': (4, 128),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-4096-sw-None': (8, 128),
                    'max_model_len-8192-sw-None': (8, 128),
                },
                'q_head-128_kv_head-2_head-256': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (16, 8),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-4_kv_head-2_head-128': {
                    'max_model_len-256-sw-None': (1, 128),
                    'max_model_len-512-sw-None': (2, 128),
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-64_kv_head-4_head-128': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-16_kv_head-1_head-128': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-32_kv_head-4_head-256': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-8_kv_head-1_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-16_kv_head-8_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (4, 64),
                    'max_model_len-4096-sw-None': (4, 64),
                    'max_model_len-8192-sw-None': (4, 64),
                },
                'q_head-4_kv_head-2_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-4096-sw-None': (8, 128),
                    'max_model_len-8192-sw-None': (8, 128),
                },
                'q_head-128_kv_head-4_head-128': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-32_kv_head-1_head-128': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-4_head-256': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (4, 32),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-16_kv_head-1_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-32_kv_head-8_head-128': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-4096-sw-None': (8, 64),
                    'max_model_len-8192-sw-None': (8, 64),
                },
                'q_head-8_kv_head-2_head-128': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 128),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-32_kv_head-2_head-128': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-1_head-128': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-128_kv_head-4_head-256': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-32_kv_head-1_head-256': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-8_head-128': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-64_kv_head-8_head-256': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (4, 16),
                },
            },
            'q_bfloat16_kv_float8_e4m3fn': {
                'q_head-2_kv_head-2_head-128': {
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-1024-sw-None': (4, 128),
                    'max_model_len-2048-sw-None': (8, 128),
                },
                'q_head-8_kv_head-4_head-128': {
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-16_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 32),
                },
                'q_head-32_kv_head-2_head-128': {
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                },
                'q_head-32_kv_head-16_head-128': {
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-8192-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (1, 32),
                },
                'q_head-64_kv_head-8_head-256': {
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                },
                'q_head-128_kv_head-2_head-256': {
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-4096-sw-None': (16, 8),
                    'max_model_len-8192-sw-None': (16, 8),
                    'max_model_len-256-sw-None': (1, 8),
                },
                'q_head-128_kv_head-8_head-128': {
                    'max_model_len-4096-sw-None': (16, 8),
                    'max_model_len-8192-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                },
                'q_head-32_kv_head-16_head-256': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (4, 16),
                },
                'q_head-64_kv_head-2_head-128': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-8_kv_head-4_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-2_kv_head-2_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 64),
                },
                'q_head-128_kv_head-8_head-256': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-4096-sw-None': (8, 8),
                    'max_model_len-8192-sw-None': (8, 8),
                },
                'q_head-32_kv_head-2_head-256': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-16_head-128': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-4_kv_head-2_head-128': {
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-512-sw-None': (2, 128),
                    'max_model_len-1024-sw-None': (4, 128),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-16_kv_head-4_head-128': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-128_kv_head-2_head-128': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-4096-sw-None': (16, 8),
                    'max_model_len-8192-sw-None': (16, 8),
                },
                'q_head-8_kv_head-2_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-16_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-32_kv_head-8_head-256': {
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-64_kv_head-8_head-128': {
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                },
                'q_head-128_kv_head-4_head-256': {
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 16),
                },
                'q_head-64_kv_head-2_head-256': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (16, 8),
                    'max_model_len-8192-sw-None': (16, 8),
                },
                'q_head-32_kv_head-4_head-128': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-128_kv_head-16_head-128': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-4096-sw-None': (8, 8),
                    'max_model_len-8192-sw-None': (8, 8),
                },
                'q_head-8_kv_head-2_head-128': {
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-16_head-256': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (4, 8),
                },
                'q_head-16_kv_head-4_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-8192-sw-None': (16, 64),
                },
                'q_head-64_kv_head-32_head-128': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (4, 8),
                },
                'q_head-4_kv_head-2_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 64),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-16_kv_head-8_head-128': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 64),
                    'max_model_len-8192-sw-None': (16, 64),
                },
                'q_head-64_kv_head-4_head-128': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-32_kv_head-4_head-256': {
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-16_kv_head-8_head-256': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-4096-sw-None': (8, 64),
                    'max_model_len-8192-sw-None': (8, 64),
                },
                'q_head-128_kv_head-4_head-128': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (16, 8),
                    'max_model_len-8192-sw-None': (16, 8),
                },
                'q_head-64_kv_head-4_head-256': {
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-32_kv_head-8_head-128': {
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
            },
        },
    },
    'TPU v5e': {
        128: {
            'q_bfloat16_kv_bfloat16': {
                'q_head-128_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-128_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (16, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-8192-sw-None': (16, 8),
                },
                'q_head-128_kv_head-16_head-128': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-128_kv_head-16_head-256': {
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (4, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (4, 8),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (4, 8),
                },
                'q_head-128_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-128_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (8, 8),
                },
                'q_head-128_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-128_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-128_kv_head-8_head-128': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-128_kv_head-8_head-256': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (4, 16),
                },
                'q_head-16_kv_head-1_head-128': {
                    'max_model_len-2048-sw-None': (8, 64),
                    512: (4, 64)
                },
                'q_head-16_kv_head-1_head-256': {
                    'max_model_len-128-sw-None': (1, 32),
                    256: (2, 8)
                },
                'q_head-16_kv_head-2_head-128': {
                    'max_model_len-128-sw-None': (1, 128),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-16_kv_head-2_head-256': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (2, 32),
                },
                'q_head-16_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (1, 64),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (4, 64),
                },
                'q_head-16_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (8, 128),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-8192-sw-None': (16, 64),
                },
                'q_head-16_kv_head-8_head-128': {
                    'max_model_len-1024-sw-None': (8, 256),
                    'max_model_len-128-sw-None': (1, 128),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (8, 64),
                    'max_model_len-512-sw-None': (4, 64),
                    'max_model_len-8192-sw-None': (4, 128),
                },
                'q_head-16_kv_head-8_head-256': {
                    'max_model_len-1024-sw-None': (8, 128),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-256-sw-None': (2, 64),
                    'max_model_len-4096-sw-None': (8, 128),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (8, 128),
                },
                'q_head-2_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (8, 128),
                    'max_model_len-128-sw-None': (1, 256),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (4, 256),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-2_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (8, 128),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (8, 128),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-8192-sw-None': (16, 64),
                },
                'q_head-32_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (1, 128),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (16, 64),
                    'max_model_len-512-sw-None': (4, 64),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-32_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-32_kv_head-16_head-128': {
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-4096-sw-None': (8, 64),
                    'max_model_len-512-sw-None': (4, 64),
                    'max_model_len-8192-sw-None': (8, 64),
                },
                'q_head-32_kv_head-16_head-256': {
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-4096-sw-None': (4, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-32_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-32_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-32_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-32_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-32_kv_head-8_head-128': {
                    'max_model_len-1024-sw-None': (8, 128),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-2048-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-512-sw-None': (4, 64),
                    'max_model_len-8192-sw-None': (4, 64),
                },
                'q_head-32_kv_head-8_head-256': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (4, 64),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (8, 64),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-8192-sw-None': (8, 64),
                },
                'q_head-4_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-256-sw-None': (1, 256),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-512-sw-None': (4, 128),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-4_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-4_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-128-sw-None': (1, 64),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-256-sw-None': (1, 256),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-4_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (8, 256),
                    'max_model_len-8192-sw-None': (8, 128),
                },
                'q_head-64_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (16, 16),
                },
                'q_head-64_kv_head-16_head-128': {
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-64_kv_head-16_head-256': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-2048-sw-None': (4, 16),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (4, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (4, 16),
                },
                'q_head-64_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-64_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (4, 8),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-64_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-64_kv_head-8_head-128': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-2048-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (2, 64),
                    'max_model_len-4096-sw-None': (4, 32),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-8_head-256': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (4, 32),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-8_kv_head-1_head-128': {
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (4, 128),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-8_kv_head-1_head-256': {
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-8_kv_head-2_head-128': {
                    'max_model_len-128-sw-None': (1, 64),
                    'max_model_len-256-sw-None': (2, 64),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-512-sw-None': (4, 64),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-8_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (8, 128),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-8192-sw-None': (8, 128),
                },
                'q_head-8_kv_head-4_head-128': {
                    'max_model_len-128-sw-None': (1, 16),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-512-sw-None': (4, 8),
                },
                'q_head-8_kv_head-4_head-256': {
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (4, 16),
                },
            }
        },
        256: {
            'q_bfloat16_kv_bfloat16': {
                'q_head-128_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (4, 8),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (8, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-128_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (4, 8),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (8, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-8192-sw-None': (8, 8),
                },
                'q_head-128_kv_head-16_head-128': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (4, 16),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (4, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (4, 16),
                },
                'q_head-128_kv_head-16_head-256': {
                    'max_model_len-1024-sw-None': (2, 8),
                    'max_model_len-2048-sw-None': (2, 8),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-8192-sw-None': (2, 8),
                },
                'q_head-128_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (8, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-128_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (4, 8),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (8, 8),
                    'max_model_len-512-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (8, 8),
                },
                'q_head-128_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (4, 16),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (4, 16),
                },
                'q_head-128_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (2, 8),
                    'max_model_len-2048-sw-None': (4, 16),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (8, 8),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-8192-sw-None': (4, 16),
                },
                'q_head-128_kv_head-8_head-128': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (4, 32),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (2, 32),
                },
                'q_head-128_kv_head-8_head-256': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (2, 16),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (2, 16),
                },
                'q_head-16_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (1, 64),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-16_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (4, 16),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-16_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (4, 64),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (8, 128),
                },
                'q_head-16_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (4, 16),
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-16_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (2, 64),
                    'max_model_len-2048-sw-None': (2, 64),
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-4096-sw-None': (4, 32),
                    'max_model_len-512-sw-None': (2, 128),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-16_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (2, 64),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (4, 128),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-16_kv_head-8_head-128': {
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (2, 128),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-8192-sw-None': (8, 128),
                },
                'q_head-16_kv_head-8_head-256': {
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (4, 128),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (4, 128),
                    'max_model_len-512-sw-None': (1, 32),
                    'max_model_len-8192-sw-None': (4, 128),
                },
                'q_head-2_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-4096-sw-None': (8, 256),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-8192-sw-None': (8, 256),
                },
                'q_head-2_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (4, 128),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (8, 256),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-32_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (4, 16),
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (8, 64),
                },
                'q_head-32_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-32_kv_head-16_head-128': {
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (4, 64),
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-4096-sw-None': (4, 64),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (4, 64),
                },
                'q_head-32_kv_head-16_head-256': {
                    'max_model_len-1024-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (2, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (2, 32),
                },
                'q_head-32_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (4, 32),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-32_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-32_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (4, 128),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (4, 128),
                },
                'q_head-32_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (2, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-32_kv_head-8_head-128': {
                    'max_model_len-1024-sw-None': (4, 128),
                    'max_model_len-2048-sw-None': (4, 128),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (4, 128),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (2, 64),
                },
                'q_head-32_kv_head-8_head-256': {
                    'max_model_len-1024-sw-None': (2, 64),
                    'max_model_len-2048-sw-None': (2, 32),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (4, 64),
                    'max_model_len-512-sw-None': (1, 32),
                    'max_model_len-8192-sw-None': (4, 64),
                },
                'q_head-4_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 128),
                    'max_model_len-4096-sw-None': (4, 128),
                    'max_model_len-512-sw-None': (2, 128),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-4_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-4096-sw-None': (8, 64),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-8192-sw-None': (4, 64),
                },
                'q_head-4_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (4, 256),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-4096-sw-None': (8, 256),
                    'max_model_len-512-sw-None': (1, 64),
                    'max_model_len-8192-sw-None': (8, 128),
                },
                'q_head-4_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (8, 64),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-8192-sw-None': (4, 64),
                },
                'q_head-64_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (2, 8),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (8, 8),
                },
                'q_head-64_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (4, 8),
                    'max_model_len-512-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-64_kv_head-16_head-128': {
                    'max_model_len-1024-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (2, 32),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-64_kv_head-16_head-256': {
                    'max_model_len-1024-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (2, 16),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (2, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (2, 16),
                },
                'q_head-64_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-64_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (2, 8),
                    'max_model_len-2048-sw-None': (4, 16),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (4, 16),
                    'max_model_len-512-sw-None': (2, 8),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-64_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-64_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (4, 16),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-64_kv_head-8_head-128': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-2048-sw-None': (2, 32),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-64_kv_head-8_head-256': {
                    'max_model_len-1024-sw-None': (4, 32),
                    'max_model_len-2048-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (1, 8),
                    'max_model_len-4096-sw-None': (4, 32),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-8_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (4, 8),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (8, 64),
                    'max_model_len-512-sw-None': (2, 32),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-8_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (8, 8),
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-4096-sw-None': (8, 64),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (8, 64),
                },
                'q_head-8_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (1, 16),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (2, 128),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-8_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (2, 32),
                    'max_model_len-256-sw-None': (1, 32),
                    'max_model_len-4096-sw-None': (4, 64),
                    'max_model_len-512-sw-None': (2, 16),
                    'max_model_len-8192-sw-None': (4, 64),
                },
                'q_head-8_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (4, 256),
                    'max_model_len-2048-sw-None': (4, 32),
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-4096-sw-None': (8, 64),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-8192-sw-None': (4, 64),
                },
                'q_head-8_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (4, 64),
                    'max_model_len-2048-sw-None': (4, 64),
                    'max_model_len-256-sw-None': (1, 64),
                    'max_model_len-4096-sw-None': (4, 128),
                    'max_model_len-512-sw-None': (2, 64),
                    'max_model_len-8192-sw-None': (4, 128),
                },
            }
        },
        64: {
            'q_bfloat16_kv_bfloat16': {
                'q_head-128_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-64-sw-None': (1, 8),
                },
                'q_head-128_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (16, 8),
                    'max_model_len-2048-sw-None': (32, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (8, 8),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (32, 8),
                },
                'q_head-128_kv_head-16_head-128': {
                    'max_model_len-1024-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-512-sw-None': (8, 16),
                    'max_model_len-64-sw-None': (1, 8),
                },
                'q_head-128_kv_head-16_head-256': {
                    'max_model_len-128-sw-None': (2, 8),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (8, 8),
                    'max_model_len-512-sw-None': (8, 8),
                    'max_model_len-64-sw-None': (1, 8),
                },
                'q_head-128_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (16, 16),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-512-sw-None': (8, 16),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (32, 16),
                },
                'q_head-128_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (16, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (32, 8),
                },
                'q_head-128_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (16, 8),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-512-sw-None': (8, 32),
                    'max_model_len-64-sw-None': (1, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-128_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (8, 8),
                    'max_model_len-128-sw-None': (2, 8),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-128_kv_head-8_head-128': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (8, 32),
                },
                'q_head-128_kv_head-8_head-256': {
                    'max_model_len-128-sw-None': (2, 8),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-16_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (16, 8),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (16, 64),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (32, 64),
                    'max_model_len-512-sw-None': (8, 16),
                    'max_model_len-64-sw-None': (1, 128),
                    'max_model_len-8192-sw-None': (32, 128),
                },
                'q_head-16_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (32, 8),
                    'max_model_len-256-sw-None': (4, 64),
                    'max_model_len-4096-sw-None': (32, 16),
                    'max_model_len-512-sw-None': (8, 8),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (32, 16),
                },
                'q_head-16_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (2, 64),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-256-sw-None': (4, 128),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-512-sw-None': (8, 64),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (32, 64),
                },
                'q_head-16_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (2, 8),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (8, 16),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-16_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-128-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (4, 128),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-512-sw-None': (4, 128),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-16_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (16, 32),
                    'max_model_len-128-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (16, 128),
                    'max_model_len-256-sw-None': (4, 32),
                    'max_model_len-4096-sw-None': (16, 128),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-16_kv_head-8_head-128': {
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-128-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-256-sw-None': (4, 64),
                    'max_model_len-4096-sw-None': (32, 64),
                    'max_model_len-512-sw-None': (8, 8),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (8, 128),
                },
                'q_head-16_kv_head-8_head-256': {
                    'max_model_len-1024-sw-None': (8, 128),
                    'max_model_len-128-sw-None': (2, 8),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-256-sw-None': (4, 32),
                    'max_model_len-4096-sw-None': (8, 128),
                    'max_model_len-512-sw-None': (8, 64),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (8, 128),
                },
                'q_head-2_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (16, 256),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (32, 32),
                    'max_model_len-256-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (32, 64),
                    'max_model_len-512-sw-None': (8, 256),
                    'max_model_len-64-sw-None': (1, 256),
                    'max_model_len-8192-sw-None': (32, 128),
                },
                'q_head-2_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-2048-sw-None': (16, 64),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-512-sw-None': (8, 32),
                    'max_model_len-8192-sw-None': (32, 64),
                },
                'q_head-32_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (32, 16),
                    'max_model_len-512-sw-None': (8, 16),
                    'max_model_len-64-sw-None': (1, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-32_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (16, 8),
                    'max_model_len-256-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-512-sw-None': (8, 16),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (32, 16),
                },
                'q_head-32_kv_head-16_head-128': {
                    'max_model_len-1024-sw-None': (16, 64),
                    'max_model_len-128-sw-None': (2, 64),
                    'max_model_len-2048-sw-None': (16, 64),
                    'max_model_len-256-sw-None': (2, 32),
                    'max_model_len-4096-sw-None': (16, 64),
                    'max_model_len-512-sw-None': (8, 32),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (16, 64),
                },
                'q_head-32_kv_head-16_head-256': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (2, 8),
                    'max_model_len-2048-sw-None': (8, 32),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (8, 32),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (4, 32),
                },
                'q_head-32_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (16, 16),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (32, 64),
                    'max_model_len-512-sw-None': (8, 32),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (32, 64),
                },
                'q_head-32_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (16, 32),
                    'max_model_len-128-sw-None': (2, 8),
                    'max_model_len-2048-sw-None': (32, 32),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-512-sw-None': (8, 32),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-32_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (1, 64),
                    'max_model_len-2048-sw-None': (32, 16),
                    'max_model_len-256-sw-None': (4, 32),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-512-sw-None': (8, 16),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-32_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-512-sw-None': (4, 16),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-32_kv_head-8_head-128': {
                    'max_model_len-1024-sw-None': (16, 32),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (2, 16),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-512-sw-None': (8, 32),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-32_kv_head-8_head-256': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (8, 64),
                    'max_model_len-256-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (16, 64),
                    'max_model_len-512-sw-None': (8, 32),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (8, 64),
                },
                'q_head-4_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (16, 32),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (32, 128),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (32, 16),
                    'max_model_len-512-sw-None': (4, 32),
                    'max_model_len-64-sw-None': (1, 32),
                    'max_model_len-8192-sw-None': (32, 128),
                },
                'q_head-4_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (16, 128),
                    'max_model_len-128-sw-None': (1, 32),
                    'max_model_len-2048-sw-None': (32, 32),
                    'max_model_len-256-sw-None': (4, 32),
                    'max_model_len-4096-sw-None': (32, 64),
                    'max_model_len-512-sw-None': (8, 64),
                    'max_model_len-64-sw-None': (1, 128),
                    'max_model_len-8192-sw-None': (32, 64),
                },
                'q_head-4_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (16, 256),
                    'max_model_len-128-sw-None': (2, 256),
                    'max_model_len-2048-sw-None': (32, 32),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (32, 64),
                    'max_model_len-512-sw-None': (8, 32),
                    'max_model_len-64-sw-None': (1, 32),
                    'max_model_len-8192-sw-None': (32, 64),
                },
                'q_head-4_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (8, 64),
                    'max_model_len-128-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (32, 128),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (32, 128),
                    'max_model_len-512-sw-None': (8, 16),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (16, 128),
                },
                'q_head-64_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (32, 16),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (32, 16),
                    'max_model_len-512-sw-None': (8, 8),
                    'max_model_len-64-sw-None': (1, 16),
                },
                'q_head-64_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (32, 8),
                    'max_model_len-256-sw-None': (2, 8),
                    'max_model_len-4096-sw-None': (32, 8),
                    'max_model_len-512-sw-None': (8, 8),
                    'max_model_len-64-sw-None': (1, 8),
                },
                'q_head-64_kv_head-16_head-128': {
                    'max_model_len-1024-sw-None': (16, 32),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-256-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (8, 32),
                    'max_model_len-512-sw-None': (8, 16),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-16_head-256': {
                    'max_model_len-1024-sw-None': (4, 16),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (8, 16),
                    'max_model_len-256-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (8, 16),
                    'max_model_len-512-sw-None': (8, 16),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (8, 16),
                },
                'q_head-64_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (32, 32),
                    'max_model_len-256-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (32, 16),
                    'max_model_len-512-sw-None': (8, 64),
                    'max_model_len-64-sw-None': (1, 32),
                },
                'q_head-64_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (16, 16),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (32, 16),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-512-sw-None': (8, 8),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (32, 16),
                },
                'q_head-64_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (8, 16),
                    'max_model_len-128-sw-None': (1, 8),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-512-sw-None': (8, 64),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (16, 16),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (16, 16),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-8_head-128': {
                    'max_model_len-1024-sw-None': (16, 64),
                    'max_model_len-128-sw-None': (2, 16),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (16, 64),
                    'max_model_len-64-sw-None': (1, 32),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-64_kv_head-8_head-256': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (2, 8),
                    'max_model_len-2048-sw-None': (16, 32),
                    'max_model_len-256-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-512-sw-None': (8, 32),
                    'max_model_len-64-sw-None': (1, 8),
                    'max_model_len-8192-sw-None': (16, 32),
                },
                'q_head-8_kv_head-1_head-128': {
                    'max_model_len-1024-sw-None': (16, 64),
                    'max_model_len-128-sw-None': (2, 64),
                    'max_model_len-2048-sw-None': (32, 32),
                    'max_model_len-256-sw-None': (4, 128),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-512-sw-None': (8, 8),
                    'max_model_len-64-sw-None': (1, 128),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-8_kv_head-1_head-256': {
                    'max_model_len-1024-sw-None': (16, 64),
                    'max_model_len-128-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (32, 32),
                    'max_model_len-256-sw-None': (4, 16),
                    'max_model_len-4096-sw-None': (32, 64),
                    'max_model_len-512-sw-None': (8, 8),
                    'max_model_len-64-sw-None': (1, 32),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-8_kv_head-2_head-128': {
                    'max_model_len-1024-sw-None': (16, 64),
                    'max_model_len-128-sw-None': (2, 64),
                    'max_model_len-2048-sw-None': (32, 32),
                    'max_model_len-256-sw-None': (4, 128),
                    'max_model_len-4096-sw-None': (32, 32),
                    'max_model_len-512-sw-None': (8, 128),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (32, 32),
                },
                'q_head-8_kv_head-2_head-256': {
                    'max_model_len-1024-sw-None': (16, 128),
                    'max_model_len-128-sw-None': (2, 64),
                    'max_model_len-2048-sw-None': (32, 32),
                    'max_model_len-256-sw-None': (4, 8),
                    'max_model_len-4096-sw-None': (16, 32),
                    'max_model_len-512-sw-None': (8, 64),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (32, 128),
                },
                'q_head-8_kv_head-4_head-128': {
                    'max_model_len-1024-sw-None': (16, 32),
                    'max_model_len-128-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (32, 64),
                    'max_model_len-256-sw-None': (4, 32),
                    'max_model_len-4096-sw-None': (16, 64),
                    'max_model_len-512-sw-None': (8, 64),
                    'max_model_len-64-sw-None': (1, 16),
                    'max_model_len-8192-sw-None': (16, 64),
                },
                'q_head-8_kv_head-4_head-256': {
                    'max_model_len-1024-sw-None': (8, 32),
                    'max_model_len-128-sw-None': (2, 32),
                    'max_model_len-2048-sw-None': (8, 128),
                    'max_model_len-256-sw-None': (4, 64),
                    'max_model_len-4096-sw-None': (8, 128),
                    'max_model_len-512-sw-None': (8, 128),
                    'max_model_len-64-sw-None': (1, 64),
                    'max_model_len-8192-sw-None': (8, 128),
                },
            }
        },
    },
}


def get_tuned_block_sizes(
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    page_size,
    max_num_tokens,
    pages_per_seq,
    sliding_window=None,
) -> tuple[int, int]:
    """Search tuned values for (num_kv_pages_per_blk, num_queries_per_blk)."""

    keys = get_lookup_keys(
        page_size,
        q_dtype,
        kv_dtype,
        actual_num_q_heads,
        actual_num_kv_heads,
        head_dim,
        page_size * pages_per_seq,
        sliding_window,
    )
    device, page_size, dtypes, head_dims, extra = keys

    try:
        bkv_p, bq = TUNED_BLOCK_SIZES[device][page_size][dtypes][head_dims][
            extra]
    except KeyError:
        logger.warning_once(
            'Couldn`t find tuned sizes for the RPA v3 kernel with %s', keys)
        # When not available use a sensible default based on TPU version
        # Set default block sizes for each tpu_version.
        tpu_version = get_tpu_version()
        if tpu_version < 4:
            raise NotImplementedError('TPU version must be 4 or higher.')
        match tpu_version:
            case 4:
                # TPUv4 has much smaller VMEM size so we pick fixed block sizes.
                bkv_p, bq = (512 // page_size, 32)
            case 7:
                bkv_p, bq = (4096 // page_size, 32)
            case _:
                bkv_p, bq = (2048 // page_size, 32)

    # We should consider the actual page_per_seq and max_num_tokens.
    # If page_per_seq < bkv_p or max_num_tokens < bq, using the bkv_p or bq may
    # waste computation. So we need the min here.
    bkv_p, bq = (min(pages_per_seq, bkv_p), min(max_num_tokens, bq))

    logger.info_once('RPA v3 kernel tuned block sizes for %s: bkv_p=%s, bq=%s',
                     keys, bkv_p, bq)
    return bkv_p, bq


def get_lookup_keys(
    page_size,
    q_dtype,
    kv_dtype,
    num_q_heads,
    num_kv_heads,
    head_dim,
    max_model_len,
    sliding_window,
):
    """Get the lookup keys for tuned block sizes."""
    (
        page_size,
        q_dtype_name,
        kv_dtype_name,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
        sliding_window,
    ) = get_simplified_raw_key(
        page_size,
        q_dtype,
        kv_dtype,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_model_len,
        sliding_window,
    )

    return (
        get_device_name(),
        next_power_of_2(page_size),
        f'q_{q_dtype_name}_kv_{kv_dtype_name}',
        f'q_head-{num_q_heads}_kv_head-{num_kv_heads}_head-{head_dim}',
        f'max_model_len-{next_power_of_2(max_model_len)}-sw-{sliding_window}',
    )


def get_simplified_raw_key(
    page_size,
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    max_model_len,
    sliding_window,
):
    """Get the simplified key."""
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads_x2 = align_to(actual_num_kv_heads * 2, kv_packing)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head,
                                       q_packing)
    assert num_kv_heads_x2 % 2 == 0

    return (
        next_power_of_2(page_size),
        jnp.dtype(q_dtype).name,
        jnp.dtype(kv_dtype).name,
        next_power_of_2(num_q_heads_per_kv_head * actual_num_kv_heads),
        next_power_of_2(num_kv_heads_x2) // 2,
        align_to(head_dim, 128),
        next_power_of_2(max_model_len),
        sliding_window,
    )
