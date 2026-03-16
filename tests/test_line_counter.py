"""Unit tests for LineCounter."""

import pytest
from src.line_counter import LineCounter


class TestLineCounterInit:
    """Test LineCounter initialization and direction mapping."""

    def test_default_init(self):
        lc = LineCounter(line_id=0, point1=(100, 200), point2=(300, 200))
        assert lc.line_id == 0
        assert lc.point1 == (100, 200)
        assert lc.point2 == (300, 200)
        assert lc.in_count == 0
        assert lc.out_count == 0

    def test_get_counts_empty(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100))
        assert lc.get_counts() == {"in": 0, "out": 0}

    def test_get_line_coords(self):
        lc = LineCounter(line_id=0, point1=(10, 20), point2=(30, 40))
        assert lc.get_line_coords() == ((10, 20), (30, 40))


class TestSideDetection:
    """Test the cross-product based side detection."""

    def test_point_above_horizontal_line(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100))
        side = lc._get_side((250, 50))
        assert side in ("positive", "negative")

    def test_point_below_horizontal_line(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100))
        above = lc._get_side((250, 50))
        below = lc._get_side((250, 150))
        assert above != below

    def test_point_on_line(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100))
        assert lc._get_side((250, 100)) == "on_line"

    def test_point_left_of_vertical_line(self):
        lc = LineCounter(line_id=0, point1=(200, 0), point2=(200, 500))
        left = lc._get_side((100, 250))
        right = lc._get_side((300, 250))
        assert left != right


class TestCrossingDetection:
    """Test line crossing detection."""

    def test_no_crossing_same_side(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100), in_direction="top")
        result = lc.check_crossing(
            track_id=1,
            prev_centroid=(250, 50),
            curr_centroid=(260, 60),
            frame_count=1
        )
        assert result is None

    def test_crossing_horizontal_line_top_to_bottom(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100), in_direction="top")
        result = lc.check_crossing(
            track_id=1,
            prev_centroid=(250, 80),
            curr_centroid=(250, 120),
            frame_count=1
        )
        assert result in ("in", "out")

    def test_crossing_horizontal_line_bottom_to_top(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100), in_direction="top")
        result = lc.check_crossing(
            track_id=2,
            prev_centroid=(250, 120),
            curr_centroid=(250, 80),
            frame_count=1
        )
        assert result in ("in", "out")

    def test_opposite_crossings_different_direction(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100), in_direction="top")
        dir1 = lc.check_crossing(1, (250, 80), (250, 120), frame_count=1)
        dir2 = lc.check_crossing(2, (250, 120), (250, 80), frame_count=1)
        assert dir1 is not None and dir2 is not None
        assert dir1 != dir2

    def test_crossing_vertical_line(self):
        lc = LineCounter(line_id=0, point1=(200, 0), point2=(200, 500), in_direction="left")
        result = lc.check_crossing(
            track_id=1,
            prev_centroid=(180, 250),
            curr_centroid=(220, 250),
            frame_count=1
        )
        assert result in ("in", "out")

    def test_count_increments(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100), in_direction="bottom")
        lc.check_crossing(1, (250, 80), (250, 120), frame_count=1)
        # Use x=100 (>20px from first crossing at x=250) to avoid spatial dedup
        lc.check_crossing(2, (100, 80), (100, 120), frame_count=2)
        counts = lc.get_counts()
        assert counts["in"] + counts["out"] == 2

    def test_no_crossing_parallel_movement(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100), in_direction="top")
        result = lc.check_crossing(
            track_id=1,
            prev_centroid=(100, 50),
            curr_centroid=(400, 50),
            frame_count=1
        )
        assert result is None


class TestCooldown:
    """Test re-entry cooldown behavior."""

    def test_cooldown_blocks_recount(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100),
                         in_direction="top", cooldown_frames=30)
        lc.check_crossing(1, (250, 80), (250, 120), frame_count=1)
        # Same track crosses back within cooldown
        result = lc.check_crossing(1, (250, 120), (250, 80), frame_count=5)
        assert result is None

    def test_cooldown_expires(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100),
                         in_direction="top", cooldown_frames=10)
        lc.check_crossing(1, (250, 80), (250, 120), frame_count=1)
        # Same track crosses back AFTER cooldown
        result = lc.check_crossing(1, (250, 120), (250, 80), frame_count=50)
        assert result is not None


class TestClassFiltering:
    """Test per-class filtering."""

    def test_allowed_class_passes(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100),
                         in_direction="top", allowed_classes=[0, 1])
        result = lc.check_crossing(1, (250, 80), (250, 120), frame_count=1, class_id=0)
        assert result is not None

    def test_disallowed_class_blocked(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100),
                         in_direction="top", allowed_classes=[0, 1])
        result = lc.check_crossing(1, (250, 80), (250, 120), frame_count=1, class_id=5)
        assert result is None

    def test_no_filter_allows_all(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100),
                         in_direction="top")
        result = lc.check_crossing(1, (250, 80), (250, 120), frame_count=1, class_id=99)
        assert result is not None


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_counts(self):
        lc = LineCounter(line_id=0, point1=(0, 100), point2=(500, 100), in_direction="top")
        lc.check_crossing(1, (250, 80), (250, 120), frame_count=1)
        lc.reset()
        assert lc.get_counts() == {"in": 0, "out": 0}
        assert len(lc.last_crossing_frames) == 0
