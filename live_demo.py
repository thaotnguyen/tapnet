# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Live Demo for Online TAPIR."""

import functools
import time

import cv2
import haiku as hk
import jax
import jax.numpy as jnp
import mediapy as media
import numpy as np
from tapnet import tapir_model
from tapnet.utils import model_utils
from tapnet.utils import transforms
from tapnet.utils import viz_utils


video = media.read_video('assets/videos/P800853_11_10_22_run3.mp4')
NUM_POINTS = 8


def load_checkpoint(checkpoint_path):
  ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
  return ckpt_state["params"], ckpt_state["state"]


tapir = lambda: tapir_model.TAPIR(
    use_causal_conv=True, bilinear_interp_with_depthwise_conv=False
)


def build_online_model_init(frames, points):
  tapir_instance = tapir()
  feature_grids = tapir_instance.get_feature_grids(frames, is_training=False)
  features = tapir_instance.get_query_features(
      frames,
      is_training=False,
      query_points=points,
      feature_grids=feature_grids,
  )
  return features

def build_online_model_init(frames, points):
  tapir_instance = tapir()
  feature_grids = tapir_instance.get_feature_grids(frames, is_training=False)
  features = tapir_instance.get_query_features(
      frames,
      is_training=False,
      query_points=points,
      feature_grids=feature_grids,
  )
  return features

def build_online_model_predict(frames, features, causal_context):
  """Compute point tracks and occlusions given frames and query points."""
  tapir_instance = tapir()
  feature_grids = tapir_instance.get_feature_grids(frames, is_training=False)
  trajectories = tapir_instance.estimate_trajectories(
      frames.shape[-3:-1],
      is_training=False,
      feature_grids=feature_grids,
      query_features=features,
      query_points_in_video=None,
      query_chunk_size=64,
      causal_context=causal_context,
      get_causal_context=True,
  )
  causal_context = trajectories["causal_context"]
  del trajectories["causal_context"]
  return {k: v[-1] for k, v in trajectories.items()}, causal_context

def sample_random_points(frame_max_idx, height, width, num_points):
  """Sample random points with (time, height, width) order."""
  y = np.random.randint(0, height, (num_points, 1))
  x = np.random.randint(0, width, (num_points, 1))
  t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
  points = np.concatenate((t, y, x), axis=-1).astype(np.int32)  # [num_points, 3]
  return points

def preprocess_frames(frames):
  """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
  frames = frames.astype(np.float32)
  frames = frames / 255 * 2 - 1
  return frames

def postprocess_occlusions(occlusions, expected_dist):
  """Postprocess occlusions to boolean visible flag.

  Args:
    occlusions: [num_points, num_frames], [-inf, inf], np.float32

  Returns:
    visibles: [num_points, num_frames], bool
  """
  pred_occ = jax.nn.sigmoid(occlusions)
  pred_occ = 1 - (1 - pred_occ) * (1 - jax.nn.sigmoid(expected_dist))
  visibles = pred_occ < 0.5  # threshold
  return visibles

def get_frame(video_capture):
  r_val, image = video_capture.read()
  trunc = np.abs(image.shape[1] - image.shape[0]) // 2
  if image.shape[1] > image.shape[0]:
    image = image[:, trunc:-trunc]
  elif image.shape[1] < image.shape[0]:
    image = image[trunc:-trunc]
  return r_val, image


print("Loading checkpoint...")
# --------------------
# Load checkpoint and initialize
params, state = load_checkpoint(
    "tapnet/checkpoints/causal_tapir_checkpoint.npy"
)

print("Creating model...")
online_init = hk.transform_with_state(build_online_model_init)
online_init_apply = jax.jit(online_init.apply)

online_predict = hk.transform_with_state(build_online_model_predict)
online_predict_apply = jax.jit(online_predict.apply)

rng = jax.random.PRNGKey(42)
online_init_apply = functools.partial(
    online_init_apply, params=params, state=state, rng=rng
)
online_predict_apply = functools.partial(
    online_predict_apply, params=params, state=state, rng=rng
)

pos = tuple()
query_frame = True
have_point = [False] * NUM_POINTS
query_features = None
causal_state = None
next_query_idx = 0

print("Compiling jax functions (this may take a while...)")
# --------------------
# Call one time to compile

height, width = video.shape[1:3]

resize_height = 256  # @param {type: "integer"}
resize_width = 256  # @param {type: "integer"}
num_points = 20  # @param {type: "integer"}

frames = media.resize_video(video, (resize_height, resize_width))
query_points = sample_random_points(0, frames.shape[1], frames.shape[2], num_points)
query_features, _ = online_init_apply(frames=preprocess_frames(frames[None, None, 0]), points=query_points[None])

causal_state = hk.transform_with_state(lambda : tapir().construct_initial_causal_state(
    NUM_POINTS, len(query_features.resolutions) - 1
)).apply(params=params, state=state, rng=rng)[0]

update_query_features_apply = functools.partial(
  hk.transform_with_state(lambda **kwargs: tapir().update_query_features(**kwargs)).apply,
  params=params, state=state, rng=rng
)

t = time.time()
step_counter = 0

predictions = []
for i in range(frames.shape[0]):
  print('{:.1}%\r'.format(float(i*100)/frames.shape[0]), end='')
  (prediction, causal_state), _ = online_predict_apply(
      frames=preprocess_frames(frames[None, None, i]),
      features=query_features,
      causal_context=causal_state,
  )
  predictions.append(prediction)

tracks = np.concatenate([x['tracks'][0] for x in predictions], axis=1)
occlusions = np.concatenate([x['occlusion'][0] for x in predictions], axis=1)
expected_dist = np.concatenate([x['expected_dist'][0] for x in predictions], axis=1)

visibles = postprocess_occlusions(occlusions, expected_dist)

# Visualize sparse point tracks
colormap = viz_utils.get_colors(20)
tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
video_viz = viz_utils.paint_point_track(video, tracks, visibles, colormap)
media.write_video('saved_video.mp4', video_viz, fps=10)
