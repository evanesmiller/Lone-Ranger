"""
Phase 1 core: coordinate normalization + tested geometry for range metrics.

Everything downstream (distance closed, closing rate, path efficiency, AOE)
depends on two things being unambiguously correct:

  1. NORMALIZATION. Plays run in both directions ('left' and 'right'). Before any
     metric is comparable across plays, every play is flipped to a canonical
     orientation: offense always moving in +x. Raw coordinates are kept too, so
     nothing is lost. A target-relative frame (origin at the ball landing point)
     is also provided, because most range questions are naturally expressed as
     "how far is the safety from where the ball is going?"

  2. GEOMETRY. Distance, distance-closed, closing rate, and path length are each
     defined once here and unit-tested against hand-computed values, so no metric
     re-implements them and inherits a silent error.

DESIGN NOTES / GUARDRAILS
  - Field is 120 (x) by 53.3 (y). Canonical flip for a 'left' play:
        x' = 120 - x ,  y' = 53.3 - y
    (rotate 180 about field center) so a left-moving offense becomes right-moving
    and lateral relationships are preserved. Applied identically to player tracks
    and to the ball landing point, so their relationship is unchanged -- the flip
    is a relabeling, and any distance computed after it equals the distance before.
    That invariant is exactly what the tests check.
  - Distance closed = D(release) - D(arrival), where D is straight-line distance
    from the safety to the ball landing point. Positive = safety got closer.
  - Closing rate = distance closed / flight_time_s (yards per second).
  - Path length = summed frame-to-frame displacement over the output window
    (actual ground covered), for path-efficiency = |net displacement| / path length.

This module computes geometry only. It makes NO judgement about whether closing
was the safety's job -- that is what role_proxy already gated upstream.
"""

import numpy as np
import pandas as pd

FIELD_X = 120.0
FIELD_Y = 53.3


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def normalize_xy(x, y, play_direction):
    """Flip 'left' plays to canonical (+x) orientation. Vectorized.

    Returns (x_norm, y_norm). 'right' plays are unchanged; 'left' plays are
    rotated 180 degrees about the field center. Applied to any point in field
    coordinates -- player or ball -- so relative geometry is preserved.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    left = np.asarray(play_direction) == "left"
    x_norm = np.where(left, FIELD_X - x, x)
    y_norm = np.where(left, FIELD_Y - y, y)
    return x_norm, y_norm


def to_target_relative(x_norm, y_norm, land_x_norm, land_y_norm):
    """Coordinates with origin at the (normalized) ball landing point.

    Positive rx = past the landing spot downfield; the magnitude hypot(rx,ry)
    is the distance to where the ball is going.
    """
    return x_norm - land_x_norm, y_norm - land_y_norm


# --------------------------------------------------------------------------
# Angle convention  (from the official NFL field diagram)
# --------------------------------------------------------------------------
# CRITICAL: dir and o are COMPASS-style, not standard math angles.
#   0 deg   -> +y  (toward the visitor sideline)
#   90 deg  -> +x  (toward the visitor endzone / increasing yardline)
#   180 deg -> -y
#   270 deg -> -x
# Angles increase CLOCKWISE. So the velocity components are:
#   vx = s * sin(dir)   vy = s * cos(dir)
# NOT the usual (cos, sin). Getting this backwards silently corrupts every
# direction-based feature, so it is defined once here and unit-tested.
def velocity_components(s, dir_deg):
    """Convert speed + compass heading (dir, degrees) into (vx, vy) on the field.

    Uses the NFL tracking convention: 0deg = +y, increasing clockwise.
    Works for a scalar or an array. Same math applies to orientation `o`
    (which way the player faces) if a facing vector is ever needed.
    """
    s = np.asarray(s, dtype=float)
    r = np.radians(np.asarray(dir_deg, dtype=float))
    vx = s * np.sin(r)
    vy = s * np.cos(r)
    return vx, vy


def normalize_angle(dir_deg, play_direction):
    """Rotate a heading to match the position flip in normalize_xy.

    A 'left' play is mirrored 180deg about field center, so every heading must
    also rotate 180deg (mod 360) to stay consistent with the flipped positions.
    'right' plays are unchanged. Without this, direction features are
    inconsistent between left and right plays.
    """
    d = np.asarray(dir_deg, dtype=float)
    left = np.asarray(play_direction) == "left"
    return np.where(left, np.mod(d + 180.0, 360.0), d)


def radial_velocity_toward(px, py, vx, vy, target_x, target_y):
    """Component of a player's velocity pointing straight at a target point.

    Positive = closing on the target; negative = moving away. This is the
    honest "is the safety already driving on the ball at release" feature:
    it projects the velocity vector onto the unit vector from player to target.
    Returns np.nan if the player is exactly on the target (direction undefined).
    """
    dx = np.asarray(target_x, dtype=float) - np.asarray(px, dtype=float)
    dy = np.asarray(target_y, dtype=float) - np.asarray(py, dtype=float)
    norm = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        ux, uy = dx / norm, dy / norm
        rv = np.asarray(vx) * ux + np.asarray(vy) * uy
    return np.where(norm > 0, rv, np.nan)


# --------------------------------------------------------------------------
# Core geometry
# --------------------------------------------------------------------------
def dist(ax, ay, bx, by):
    """Euclidean distance between point(s) A and B."""
    return np.hypot(np.asarray(ax) - np.asarray(bx),
                    np.asarray(ay) - np.asarray(by))


def distance_closed(release_x, release_y, arrival_x, arrival_y, land_x, land_y):
    """D(release -> land) - D(arrival -> land). Positive = closed on the ball."""
    d_release = dist(release_x, release_y, land_x, land_y)
    d_arrival = dist(arrival_x, arrival_y, land_x, land_y)
    return d_release - d_arrival


def closing_rate(dist_closed, flight_time_s):
    """Yards closed per second of ball flight."""
    ft = np.asarray(flight_time_s, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.asarray(dist_closed) / ft
    return np.where(ft > 0, rate, np.nan)


def path_length(xs, ys):
    """Total ground covered along an ordered track (sum of frame-to-frame steps)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 2:
        return 0.0
    return float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))


def path_efficiency(xs, ys):
    """Net displacement / ground covered. 1.0 = perfectly straight; ->0 = wandering.

    Returns np.nan for a degenerate track (no movement), so callers can filter
    rather than divide by zero.
    """
    pl = path_length(xs, ys)
    if pl == 0:
        return np.nan
    net = np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    return net / pl


# --------------------------------------------------------------------------
# Trajectory assembly (joins frozen opportunities to output tracking)
# --------------------------------------------------------------------------
def build_trajectory(opp_row, output_df):
    """Return the safety's post-throw track for one frozen opportunity, in both
    raw and normalized coordinates, ordered by frame.

    opp_row: a row (Series) from the frozen opportunity table.
    output_df: the output tracking for the relevant week (columns game_id,
               play_id, nfl_id, frame_id, x, y).
    """
    t = output_df[(output_df.game_id == opp_row.game_id) &
                  (output_df.play_id == opp_row.play_id) &
                  (output_df.nfl_id == opp_row.nfl_id)].sort_values("frame_id")
    if t.empty:
        return None
    xn, yn = normalize_xy(t.x.values, t.y.values, opp_row.play_direction)
    land_xn, land_yn = normalize_xy(opp_row.ball_land_x, opp_row.ball_land_y,
                                    opp_row.play_direction)
    return {
        "frame_id": t.frame_id.values,
        "x_raw": t.x.values, "y_raw": t.y.values,
        "x": xn, "y": yn,
        "land_x": float(land_xn), "land_y": float(land_yn),
    }
