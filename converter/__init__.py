# Copyright 2026 The nanochat-jp authors.
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
"""NanoChatJP: standalone HF custom model for nanochat-jp checkpoints.

The converted output directory is self-contained (auto_map + copied code), so
normally you just do:

    AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)

Importing this package directly also works when the directory is on sys.path.
"""

from .configuration_nanochat_jp import NanoChatJPConfig
from .modeling_nanochat_jp import (
    NanoChatJPForCausalLM,
    NanoChatJPModel,
    NanoChatJPPreTrainedModel,
)

__all__ = [
    "NanoChatJPConfig",
    "NanoChatJPForCausalLM",
    "NanoChatJPModel",
    "NanoChatJPPreTrainedModel",
]
