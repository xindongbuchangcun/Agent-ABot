# agentnav/core/s2_client.py
# DEPRECATED — no longer used.
#
# Originally planned as an HTTP client for a separate VLM server (Qwen3-VL).
# Replaced by the agent's native multimodal vision:
#   robot_capture() → ImageContent → agent perceives directly
#   pixel_to_pose(u, v) → pose  (via skills/locate.md)
#
# Safe to delete once confirmed nothing imports this file.
