# Official mAP Global scoring code (Kaggle notebook davidemaz/map-global-metric-code, version of 2026-10-15), kept verbatim for cross-checking.
"""
VAM Dataset Challenge Metric for Kaggle

This metric evaluates action detection predictions for the RetailAction dataset using:
- Temporal IoU (Intersection over Union) for action timing
- Spatial distance in meters for action localization
- Average Precision (AP) computed across multiple thresholds

The metric returns a global mean Average Precision (mAPglobal) score that combines
spatial and temporal evaluation criteria.

Expected DataFrame format:
- Row ID column: video_id (string)
- 'prediction' column: JSON string containing prediction data
- 'groundtruth' column: JSON string containing groundtruth data

Prediction JSON format:
{
  "actions": [
    {
      "score": 0.8,  # Confidence score
      "start": 0.0,  # Normalized time [0, 1]
      "end": 0.3,    # Normalized time [0, 1]
      "spatial": {
        "action_cam": {
          "rank0": {"x": 0.5, "y": 0.5, "visible": true}
        }
      }
    }
  ]
}

Groundtruth JSON format:
{
  "content": {
    "action_cam": {
      "rank0": {
        "poses": [...]  # Pose data for computing meter/pixel conversion
      }
    },
    "m_px_factors": {"rank0": 3.07, ...}  # Optional precomputed factors
  },
  "labels": {
    "action": [
      {
        "label": "take",
        "start": 0.0,
        "end": 0.4,
        "spatial": {
          "action_cam": {
            "rank0": {"x": 0.5, "y": 0.5}
          }
        }
      }
    ]
  }
}
"""

import json
import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, NamedTuple, Optional, Union

import numpy as np
import pandas as pd

# Configure logging only if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ParticipantVisibleError(Exception):
    """Error messages shown to competition participants."""
    pass


# ============================================================================
# Data Types (embedded to avoid external file dependencies)
# ============================================================================

class ActionPoint(NamedTuple):
    """Represents a spatial point for an action."""
    x: float  # spatial dimension: normalized [0,1] wrt video size
    y: float  # spatial dimension: normalized [0,1] wrt video size
    visible: Union[float, bool] = True  # probability of the point being visible in the view
    m_px_factor: Optional[float] = None  # meter / pixel conversion factor


class ActionEnum(IntEnum):
    """Enumeration of action types."""
    take = 0
    put = 1
    touch = 2
    other = 3
    none = 4  # Null class (means no action)

    @classmethod
    def from_raw_label(cls, label: str) -> "ActionEnum":
        label = label.lower()
        if "take" in label:
            return ActionEnum.take
        elif "put" in label:
            return ActionEnum.put
        elif "touch" in label:
            return ActionEnum.touch
        else:
            return ActionEnum.other


class ActionLocationEnum(IntEnum):
    """Enumeration of action locations."""
    null = -1
    other = 0
    cart = 1


@dataclass
class ActionPrediction:
    """Represents a predicted action."""
    probs: Union[List[float], float]  # List[float] for multi-class or float for single score
    start: float  # temporal dimension: normalized [0,1] wrt video length
    end: float  # temporal dimension: normalized [0,1] wrt video length
    spatial: Dict[str, Dict[str, ActionPoint]] = field(
        default_factory=lambda: {"action_cam": dict()}
    )


@dataclass
class Prediction:
    """Container for all predictions for a video."""
    interaction: Optional[float] = None
    actions: List[ActionPrediction] = field(default_factory=list)


@dataclass
class ActionLabel:
    """Represents a ground truth action label."""
    label: ActionEnum
    start: float  # temporal dimension: normalized [0,1] wrt video length
    end: float  # temporal dimension: normalized [0,1] wrt video length
    spatial: Dict[str, Dict[str, ActionPoint]] = field(
        default_factory=lambda: {"action_cam": dict()}
    )
    extends: bool = False  # if the action length spans outside the video time
    location: ActionLocationEnum = ActionLocationEnum.null  # location of the action

    @classmethod
    def from_dict(cls, data: Dict) -> "ActionLabel":
        spatial_raw = data["spatial"]
        spatial = {
            "action_cam": dict(),
        }
        if spatial_raw is not None:
            if "action_cam" in spatial_raw:
                spatial["action_cam"] = {
                    rank: ActionPoint(**{**point, "visible": True})
                    for rank, point in spatial_raw["action_cam"].items()
                }
        return cls(
            label=ActionEnum[data["label"]],
            start=data["start"],
            end=data["end"],
            spatial=spatial,
            extends=data.get("extends", None),
            location=ActionLocationEnum[data.get("location", "null")],
        )


@dataclass
class RawTarget:
    """Container for ground truth data."""
    classification_label: Optional[List] = None
    action_labels: Optional[List[ActionLabel]] = field(default_factory=list)
    keypoints: Dict = None
    shelves: Dict = None
    poses: Dict = None
    m_px_factors: Optional[Dict[str, float]] = None  # Precomputed meters/pixel factors per rank

# Average bone lengths in meters for pose-based meter/pixel conversion
BONE_LENGTH_MEANS = {
    ("neck", "nose"): 0.19354,
    ("left_shoulder", "left_elbow"): 0.27096,
    ("left_elbow", "left_wrist"): 0.21228,
    ("right_shoulder", "right_elbow"): 0.27210,
    ("right_elbow", "right_wrist"): 0.21316,
    ("left_hip", "left_knee"): 0.39204,
    ("left_knee", "left_ankle"): 0.39530,
    ("right_hip", "right_knee"): 0.39266,
    ("right_knee", "right_ankle"): 0.39322,
    ("left_shoulder", "right_shoulder"): 0.35484,
    ("left_hip", "right_hip"): 0.17150,
    ("neck", "left_shoulder"): 0.18136,
    ("neck", "right_shoulder"): 0.18081,
    ("left_shoulder", "left_hip"): 0.51375,
    ("right_shoulder", "right_hip"): 0.51226,
}

M_PX_FACTOR_AVG = 3.07


def compute_m_px_factor_from_bones(raw_target: RawTarget) -> Optional[Dict]:
    """Compute meters/pixel factors based on precomputed bone lengths."""
    m_px_factors = dict()
    for rank_name, rank_poses in raw_target.poses.items():
        m_px_factors[rank_name] = []
        for pose in rank_poses:
            if pose is None:
                continue
            for (first_bone, second_bone), bone_length_meters in BONE_LENGTH_MEANS.items():
                first_joint = pose.get(first_bone)
                second_joint = pose.get(second_bone)
                
                if first_joint is not None and second_joint is not None:
                    bone_length_pixels = np.sqrt(
                        np.sum((first_joint - second_joint) ** 2)
                    )
                    if bone_length_pixels > 0:
                        m_px_factors[rank_name].append(bone_length_meters / bone_length_pixels)
    
    if len(m_px_factors) == 0:
        return None
    else:
        output = {rank_name: np.nanmedian(factors) for rank_name, factors in m_px_factors.items()}
        output = {
            rank_name: M_PX_FACTOR_AVG if np.isnan(factor) else factor
            for rank_name, factor in output.items()
        }
        return output


def parse_prediction_from_json(pred_json: str) -> Prediction:
    """Parse prediction from JSON string."""
    pred_data = json.loads(pred_json)
    actions = []
    
    for action_dict in pred_data.get("actions", []):
        spatial = {"action_cam": {}}
        # A null "spatial" or "action_cam" means no location: a spatial miss
        spatial_raw = action_dict.get("spatial") or {}
        for rank, point in (spatial_raw.get("action_cam") or {}).items():
            point = ActionPoint(**point)
            if not all(isinstance(v, (int, float)) for v in (point.x, point.y)):
                raise ParticipantVisibleError(
                    f"Spatial point '{rank}' must have numeric 'x' and 'y' "
                    "(omit the rank if there is no location)"
                )
            spatial["action_cam"][rank] = point
        
        # Require "score" field with single float value
        if "score" not in action_dict:
            raise ParticipantVisibleError("Action must have a 'score' field with a single float value")
        
        score = action_dict["score"]
        if not isinstance(score, (int, float)):
            raise ParticipantVisibleError(f"Action 'score' must be a number, got {type(score).__name__}")
        
        action_pred = ActionPrediction(
            probs=float(score),  # Store as float in probs field for compatibility
            start=action_dict["start"],
            end=action_dict["end"],
            spatial=spatial
        )
        actions.append(action_pred)
    
    return Prediction(actions=actions)


def parse_groundtruth_from_json(gt_json: str) -> RawTarget:
    """Parse groundtruth from JSON string."""
    gt_data = json.loads(gt_json)
    
    # Load poses
    poses = {}
    content = gt_data.get("content", gt_data)
    for rank, rank_dict in content.get("action_cam", {}).items():
        poses_data = rank_dict.get("poses", [])
        poses[rank] = []
        for frame_content in poses_data:
            if frame_content is not None and len(frame_content.get("pose", {})) > 0:
                pose_data = frame_content["pose"]
                joint_scores = frame_content.get("joint_scores", {})
                
                pose_dict = {}
                for joint_name, coords in pose_data.items():
                    score = joint_scores.get(joint_name, 0.0)
                    coords_array = np.array(coords)
                    if score > 0 or np.any(coords_array != 0):
                        pose_dict[joint_name] = coords_array
                
                poses[rank].append(pose_dict if pose_dict else None)
            else:
                poses[rank].append(None)
    
    # Load action labels
    labels = content.get("labels", {})
    raw_action_labels = labels.get("action", None)
    
    action_labels = []
    if raw_action_labels is not None:
        for action_label_dict in raw_action_labels:
            action_labels.append(ActionLabel.from_dict(action_label_dict))
    
    # Load precomputed m_px_factors
    m_px_factors = content.get("m_px_factors", None)
    
    return RawTarget(
        poses=poses,
        action_labels=action_labels,
        shelves={},
        m_px_factors=m_px_factors
    )


def compute_metric(
    predictions: List[Prediction],
    targets: List[RawTarget],
    confidence_thresholds: np.ndarray = np.linspace(1, 0, 101),
    tiou_thresholds: np.ndarray = np.linspace(0.05, 0.5, 10),
    spatial_thresholds: np.ndarray = np.linspace(1, 0.05, 20),
) -> float:
    """Compute mAPglobal metric.
    
    This is the core metric computation adapted from GlobalMetric.compute().
    """
    eps = 1e-4
    
    # Initialize counters
    tp = np.zeros((len(spatial_thresholds), len(tiou_thresholds), len(confidence_thresholds)))
    fp = np.zeros((len(spatial_thresholds), len(tiou_thresholds), len(confidence_thresholds)))
    fn = np.zeros((len(spatial_thresholds), len(tiou_thresholds), len(confidence_thresholds)))
    
    # Process each video
    for pred, raw_target in zip(predictions, targets):
        # Compute or use precomputed m_px_factors
        if raw_target.m_px_factors is not None:
            m_px_factors = raw_target.m_px_factors
        else:
            m_px_factors = compute_m_px_factor_from_bones(raw_target)
        
        if m_px_factors is None or any(np.isnan(factor) for factor in m_px_factors.values()):
            continue
        
        pred_actions = pred.actions
        target_actions = raw_target.action_labels
        
        if not target_actions:
            continue
        
        # Helper function to get confidence score (always a single float)
        def get_confidence(action_pred: ActionPrediction) -> float:
            # probs field always contains a single float for this metric
            return float(action_pred.probs)
        
        # Sort predictions by confidence
        pred_actions = sorted(pred_actions, key=get_confidence, reverse=True)
        
        # Iterate through thresholds
        for s_idx, spatial_thresh in enumerate(spatial_thresholds):
            for t_idx, tiou_thresh in enumerate(tiou_thresholds):
                for c_idx, conf_thresh in enumerate(confidence_thresholds):
                    matched_targets = set()
                    tp_count = fp_count = 0
                    
                    for pred_action in pred_actions:
                        confidence_score = get_confidence(pred_action)
                        if confidence_score < conf_thresh:
                            continue
                        
                        best_iou = 0
                        best_target_idx = -1
                        
                        for target_idx, target in enumerate(target_actions):
                            if target_idx in matched_targets:
                                continue
                            
                            # Compute temporal IoU
                            t_intersection = min(pred_action.end, target.end) - max(
                                pred_action.start, target.start
                            )
                            t_union = max(pred_action.end, target.end) - min(
                                pred_action.start, target.start
                            )
                            tiou = max(0, t_intersection / (t_union + eps))
                            
                            # Check spatial match: every camera shared with the target
                            # needs a rank within spatial_thresh, and at least one camera
                            # must be shared. Ranks the target lacks and non-finite
                            # distances (e.g. NaN coordinates) are misses.
                            spatial_match = False
                            for cam, ranks in pred_action.spatial.items():
                                if cam not in target.spatial:
                                    continue
                                spatial_match = False
                                for rank, pred_point in ranks.items():
                                    if rank not in target.spatial[cam] or rank not in m_px_factors:
                                        continue
                                    target_point = target.spatial[cam][rank]
                                    dist_pixels = (
                                        (pred_point.x - target_point.x) ** 2
                                        + (pred_point.y - target_point.y) ** 2
                                    ) ** 0.5
                                    dist_meters = dist_pixels * m_px_factors[rank]
                                    if math.isfinite(dist_meters) and dist_meters <= spatial_thresh:
                                        spatial_match = True
                                        break
                                if not spatial_match:
                                    break
                            
                            if tiou >= tiou_thresh and spatial_match:
                                if tiou > best_iou:
                                    best_iou = tiou
                                    best_target_idx = target_idx
                        
                        if best_target_idx >= 0:
                            tp_count += 1
                            matched_targets.add(best_target_idx)
                        else:
                            fp_count += 1
                    
                    fn_count = len(target_actions) - len(matched_targets)
                    
                    tp[s_idx, t_idx, c_idx] += tp_count
                    fp[s_idx, t_idx, c_idx] += fp_count
                    fn[s_idx, t_idx, c_idx] += fn_count
    
    # Compute AP
    precisions = tp / (tp + fp + eps)
    recalls = tp / (tp + fn + eps)
    
    # Compute maximum precision for each recall level
    for s_idx in range(len(spatial_thresholds)):
        for t_idx in range(len(tiou_thresholds)):
            precisions[s_idx, t_idx] = np.flip(
                np.maximum.accumulate(np.flip(precisions[s_idx, t_idx]))
            )
    
    # Calculate AP values
    recall_diffs = recalls[..., 1:] - recalls[..., :-1]
    ap_values = np.sum(recall_diffs * precisions[..., :-1], axis=-1)
    
    # Global mAP is mean of all values
    mAP_global = np.mean(ap_values)
    
    return float(mAP_global)


def metric(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "id",
    confidence_thresholds: np.ndarray = None,
    tiou_thresholds: np.ndarray = None,
    spatial_thresholds: np.ndarray = None,
) -> float:
    """
    Compute the VAM Dataset Challenge metric (mAPglobal).
    
    This metric evaluates action detection in retail surveillance videos using:
    - Temporal IoU for action timing accuracy
    - Spatial distance (in meters) for action localization accuracy
    - Average Precision computed across multiple threshold combinations
    
    Args:
        solution: DataFrame with groundtruth data containing:
            - Row ID column (default: "id")
            - 'groundtruth' column with JSON strings
        submission: DataFrame with prediction data containing:
            - Row ID column (default: "id")
            - 'prediction' column with JSON strings
        row_id_column_name: Name of the row ID column (default: "id")
        confidence_thresholds: Array of confidence thresholds (default: 101 values from 1 to 0)
        tiou_thresholds: Array of temporal IoU thresholds (default: 10 values from 0.05 to 0.5)
        spatial_thresholds: Array of spatial distance thresholds in meters (default: 20 values from 1 to 0.05)
    
    Returns:
        mAPglobal: Single float value representing the global mean Average Precision
    
    Raises:
        ParticipantVisibleError: For validation errors that participants should see
    
    Example:
        >>> import pandas as pd
        >>> import json
        >>> # Create simple test data
        >>> solution_data = [{
        ...     'id': 'video_1',
        ...     'groundtruth': json.dumps({
        ...         "content": {
        ...             "action_cam": {"rank0": {"poses": []}},
        ...             "m_px_factors": {"rank0": 3.0},
        ...             "labels": {
        ...                 "action": [{
        ...                     "label": "take",
        ...                     "start": 0.0,
        ...                     "end": 0.4,
        ...                     "spatial": {"action_cam": {"rank0": {"x": 0.5, "y": 0.5}}}
        ...                 }]
        ...             }
        ...         }
        ...     })
        ... }]
        >>> submission_data = [{
        ...     'id': 'video_1',
        ...     'prediction': json.dumps({
        ...         "actions": [{
        ...             "score": 0.8,
        ...             "start": 0.0,
        ...             "end": 0.5,
        ...             "spatial": {"action_cam": {"rank0": {"x": 0.5, "y": 0.5, "visible": True}}}
        ...         }]
        ...     })
        ... }]
        >>> solution_df = pd.DataFrame(solution_data)
        >>> submission_df = pd.DataFrame(submission_data)
        >>> score = metric(solution_df, submission_df)
        >>> assert 0.0 <= score <= 1.0
        >>> print(f"Score: {score:.4f}")  # doctest: +ELLIPSIS
        Score: 0...
    """
    # Set default thresholds if not provided
    if confidence_thresholds is None:
        confidence_thresholds = np.linspace(1, 0, 101)
    if tiou_thresholds is None:
        tiou_thresholds = np.linspace(0.05, 0.5, 10)
    if spatial_thresholds is None:
        spatial_thresholds = np.linspace(1, 0.05, 20)
    
    # Validate that required columns exist
    if 'prediction' not in submission.columns:
        raise ParticipantVisibleError("Submission must have a 'prediction' column")
    if 'groundtruth' not in solution.columns:
        raise ParticipantVisibleError("Solution must have a 'groundtruth' column")
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"Submission must have a '{row_id_column_name}' column")
    if row_id_column_name not in solution.columns:
        raise ParticipantVisibleError(f"Solution must have a '{row_id_column_name}' column")
    
    # Check that submission and solution have same video IDs
    solution_ids = set(solution[row_id_column_name])
    submission_ids = set(submission[row_id_column_name])
    
    if not submission_ids.issubset(solution_ids):
        extra_ids = submission_ids - solution_ids
        raise ParticipantVisibleError(
            f"Submission contains {len(extra_ids)} video IDs not in groundtruth"
        )
    
    if submission_ids != solution_ids:
        missing_ids = solution_ids - submission_ids
        raise ParticipantVisibleError(
            f"Submission is missing {len(missing_ids)} video IDs from groundtruth"
        )
    
    # Align dataframes by row_id
    solution = solution.sort_values(row_id_column_name).reset_index(drop=True)
    submission = submission.sort_values(row_id_column_name).reset_index(drop=True)
    
    # Parse predictions and groundtruth
    predictions = []
    targets = []
    
    for idx in range(len(submission)):
        video_id = submission.loc[idx, row_id_column_name]
        
        try:
            # Parse prediction
            pred_json = submission.loc[idx, 'prediction']
            pred = parse_prediction_from_json(pred_json)
            predictions.append(pred)
            
            # Parse groundtruth
            gt_json = solution.loc[idx, 'groundtruth']
            gt = parse_groundtruth_from_json(gt_json)
            targets.append(gt)
            
        except json.JSONDecodeError as e:
            raise ParticipantVisibleError(f"Invalid JSON format for video_id {video_id}: {e}")
        except KeyError as e:
            raise ParticipantVisibleError(f"Missing required field for video_id {video_id}: {e}")
        except Exception as e:
            raise ParticipantVisibleError(f"Error processing video_id {video_id}: {e}")
    
    # Compute metric
    try:
        mAP_global = compute_metric(
            predictions=predictions,
            targets=targets,
            confidence_thresholds=confidence_thresholds,
            tiou_thresholds=tiou_thresholds,
            spatial_thresholds=spatial_thresholds,
        )
    except Exception as e:
        # Don't expose internal errors to participants
        try:
            logger.error(f"Error computing metric: {e}")
        except:
            pass  # Ignore logging errors
        raise Exception(f"Internal error computing metric")
    
    # Validate output
    if not np.isfinite(mAP_global):
        raise Exception("Metric returned non-finite value")
    
    return float(mAP_global)


# Backward compatibility alias
def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    confidence_thresholds: Optional[np.ndarray] = None,
    tiou_thresholds: Optional[np.ndarray] = None,
    spatial_thresholds: Optional[np.ndarray] = None,
) -> float:
    """
    Backward compatibility alias for metric() function.
    Use metric() instead for Kaggle compatibility.
    """
    return metric(
        solution=solution,
        submission=submission,
        row_id_column_name=row_id_column_name,
        confidence_thresholds=confidence_thresholds,
        tiou_thresholds=tiou_thresholds,
        spatial_thresholds=spatial_thresholds,
    )


# Simple self-test to validate the metric works (Kaggle may run this)
if __name__ == "__main__":
    # Create minimal test case
    test_prediction = {
        "actions": [{
            "score": 0.8,
            "start": 0.0,
            "end": 0.5,
            "spatial": {"action_cam": {"rank0": {"x": 0.5, "y": 0.5, "visible": True}}}
        }]
    }
    
    test_groundtruth = {
        "content": {
            "action_cam": {"rank0": {"poses": []}},
            "m_px_factors": {"rank0": 3.0},
            "labels": {
                "action": [{
                    "label": "take",
                    "start": 0.0,
                    "end": 0.4,
                    "spatial": {"action_cam": {"rank0": {"x": 0.5, "y": 0.5}}}
                }]
            }
        }
    }
    
    submission_df = pd.DataFrame([{
        'id': 'test_1',
        'prediction': json.dumps(test_prediction)
    }])
    
    solution_df = pd.DataFrame([{
        'id': 'test_1',
        'groundtruth': json.dumps(test_groundtruth)
    }])
    
    try:
        # Test with default row_id_column_name
        result = metric(solution_df, submission_df)
        print(f"Self-test passed. mAPglobal: {result:.6f}")
        assert isinstance(result, float), "Result must be a float"
        assert np.isfinite(result), "Result must be finite"
    except Exception as e:
        print(f"Self-test failed: {e}")
        raise
