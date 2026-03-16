"""Unit tests for ROICongestion."""

import pytest
import numpy as np
from src.roi_congestion import ROICongestion


@pytest.fixture
def square_roi():
    """A simple square ROI from (100,100) to (300,300)."""
    vertices = [(100, 100), (300, 100), (300, 300), (100, 300)]
    return ROICongestion(roi_id=0, vertices=vertices, low_threshold=3, high_threshold=8)


@pytest.fixture
def dwell_roi():
    """A square ROI with dwell tracking enabled."""
    vertices = [(100, 100), (300, 100), (300, 300), (100, 300)]
    return ROICongestion(
        roi_id=0, vertices=vertices,
        enable_dwell=True, dwell_threshold=5.0
    )


class TestPointInside:
    def test_point_inside(self, square_roi):
        assert square_roi.is_point_inside((200, 200)) is True

    def test_point_outside(self, square_roi):
        assert square_roi.is_point_inside((50, 50)) is False

    def test_point_on_boundary(self, square_roi):
        assert square_roi.is_point_inside((100, 100)) is True

    def test_point_far_outside(self, square_roi):
        assert square_roi.is_point_inside((500, 500)) is False


class TestCongestionLevel:
    def test_low(self, square_roi):
        assert square_roi.get_congestion_level(1) == "Low"

    def test_medium(self, square_roi):
        assert square_roi.get_congestion_level(5) == "Medium"

    def test_high(self, square_roi):
        assert square_roi.get_congestion_level(10) == "High"

    def test_at_low_threshold(self, square_roi):
        assert square_roi.get_congestion_level(3) == "Medium"

    def test_at_high_threshold(self, square_roi):
        assert square_roi.get_congestion_level(8) == "Medium"


class TestCountObjects:
    def test_count_inside(self, square_roi):
        centroids = [((200, 200), 0), ((150, 150), 1)]
        count = square_roi.count_objects(centroids)
        assert count == 2

    def test_count_mixed(self, square_roi):
        centroids = [((200, 200), 0), ((50, 50), 1), ((250, 250), 2)]
        count = square_roi.count_objects(centroids)
        assert count == 2  # Only (200,200) and (250,250) inside

    def test_count_none_inside(self, square_roi):
        centroids = [((50, 50), 0), ((400, 400), 1)]
        count = square_roi.count_objects(centroids)
        assert count == 0

    def test_updates_level(self, square_roi):
        centroids = [((200, 200), i) for i in range(10)]
        square_roi.count_objects(centroids)
        assert square_roi.current_level == "High"


class TestClassFiltering:
    def test_allowed_class(self):
        vertices = [(100, 100), (300, 100), (300, 300), (100, 300)]
        roi = ROICongestion(roi_id=0, vertices=vertices, allowed_classes=[0, 1])
        centroids = [((200, 200), 0), ((200, 200), 5)]
        count = roi.count_objects(centroids)
        assert count == 1  # Only class 0 counted

    def test_no_filter_allows_all(self):
        vertices = [(100, 100), (300, 100), (300, 300), (100, 300)]
        roi = ROICongestion(roi_id=0, vertices=vertices, allowed_classes=None)
        centroids = [((200, 200), 0), ((200, 200), 99)]
        count = roi.count_objects(centroids)
        assert count == 2


class TestDwellTime:
    def test_dwell_disabled_returns_empty(self, square_roi):
        track_centroids = {1: ((200, 200), 0)}
        dwell_times, events = square_roi.update_dwell_times(track_centroids, current_frame=1, fps=30.0)
        assert dwell_times == {}
        assert events == []

    def test_dwell_tracks_entry(self, dwell_roi):
        track_centroids = {1: ((200, 200), 0)}
        dwell_roi.update_dwell_times(track_centroids, current_frame=0, fps=30.0)
        assert 1 in dwell_roi.dwell_entry_frames

    def test_dwell_accumulates_time(self, dwell_roi):
        track_centroids = {1: ((200, 200), 0)}
        dwell_roi.update_dwell_times(track_centroids, current_frame=0, fps=30.0)
        dwell_times, _ = dwell_roi.update_dwell_times(track_centroids, current_frame=90, fps=30.0)
        assert 1 in dwell_times
        assert dwell_times[1] == pytest.approx(3.0, abs=0.1)  # 90 frames / 30 fps

    def test_dwell_object_leaves(self, dwell_roi):
        # Object enters
        dwell_roi.update_dwell_times({1: ((200, 200), 0)}, current_frame=0, fps=30.0)
        dwell_roi.update_dwell_times({1: ((200, 200), 0)}, current_frame=60, fps=30.0)
        # Object leaves (not in centroids and centroid dict is empty)
        _, events = dwell_roi.update_dwell_times({}, current_frame=90, fps=30.0)
        assert len(events) == 1
        assert events[0]['track_id'] == 1
        assert events[0]['duration'] == pytest.approx(2.0, abs=0.1)

    def test_dwell_exceeds_threshold(self, dwell_roi):
        track_centroids = {1: ((200, 200), 0)}
        dwell_roi.update_dwell_times(track_centroids, current_frame=0, fps=30.0)
        dwell_times, _ = dwell_roi.update_dwell_times(track_centroids, current_frame=180, fps=30.0)
        assert dwell_times[1] > dwell_roi.dwell_threshold
        assert 1 in dwell_roi.get_objects_exceeding_threshold()


class TestIDRecovery:
    def test_id_recovery_transfers_dwell(self, dwell_roi):
        # Track 1 enters
        dwell_roi.update_dwell_times({1: ((200, 200), 0)}, current_frame=0, fps=30.0)
        dwell_roi.update_dwell_times({1: ((200, 200), 0)}, current_frame=30, fps=30.0)

        # Track 1 disappears, track 2 appears at same position
        dwell_roi.update_dwell_times({2: ((202, 202), 0)}, current_frame=31, fps=30.0)

        # Track 2 should have inherited track 1's dwell entry frame
        assert 2 in dwell_roi.dwell_entry_frames
        assert dwell_roi.dwell_entry_frames[2] == 0  # Original entry frame

    def test_id_recovery_different_class_no_transfer(self, dwell_roi):
        # Track 1 (class 0) enters
        dwell_roi.update_dwell_times({1: ((200, 200), 0)}, current_frame=0, fps=30.0)
        dwell_roi.update_dwell_times({1: ((200, 200), 0)}, current_frame=30, fps=30.0)

        # Track 2 (class 5, different class) appears at same position
        dwell_roi.update_dwell_times({2: ((202, 202), 5)}, current_frame=31, fps=30.0)

        # Should NOT recover — different class
        assert dwell_roi.dwell_entry_frames[2] == 31


class TestDrawOnFrame:
    def test_draw_on_frame_returns_frame(self, square_roi):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        result = square_roi.draw_on_frame(frame)
        assert result.shape == (400, 400, 3)

    def test_draw_modifies_frame(self, square_roi):
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        original = frame.copy()
        square_roi.count_objects([((200, 200), 0)])
        square_roi.draw_on_frame(frame)
        assert not np.array_equal(frame, original)


class TestGetStats:
    def test_stats_structure(self, square_roi):
        stats = square_roi.get_stats()
        assert "roi_id" in stats
        assert "name" in stats
        assert "count" in stats
        assert "level" in stats
        assert "vertices" in stats
