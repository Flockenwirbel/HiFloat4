"""Quick check of diffusers Wan classes."""
from diffusers import WanPipeline
import inspect
print("WanPipeline file:", inspect.getfile(WanPipeline))
try:
    from diffusers.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V
    print("WanPipelineI2V exists:", inspect.getfile(WanPipelineI2V))
except ImportError as e:
    print("WanPipelineI2V not found:", e)
import diffusers
print("\nAll Wan-related exports:")
for x in dir(diffusers):
    if 'wan' in x.lower():
        print(" ", x)