"""Unit tests for SyncBuffer."""

import pytest
import numpy as np
from src.sync_buffer import SyncBuffer


@pytest.fixture
def small_buffer():
    """A buffer with small size for testing (5 frames)."""
    return SyncBuffer(num_backends=2, buffer_size=5)


@pytest.fixture
def dummy_frame():
    """A small dummy frame for testing."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestAddFrame:
    def test_add_frame(self, small_buffer, dummy_frame):
        small_buffer.add_frame(0, dummy_frame)
        assert 0 in small_buffer.pending
        assert small_buffer.pending[0]['frame_id'] == 0

    def test_add_duplicate_frame_ignored(self, small_buffer, dummy_frame):
        small_buffer.add_frame(0, dummy_frame)
        small_buffer.add_frame(0, dummy_frame)
        assert len(small_buffer.pending) == 1

    def test_add_multiple_frames(self, small_buffer, dummy_frame):
        for i in range(5):
            small_buffer.add_frame(i, dummy_frame)
        assert len(small_buffer.pending) == 5


class TestAddResults:
    def test_add_results_to_existing_frame(self, small_buffer, dummy_frame):
        small_buffer.add_frame(0, dummy_frame)
        detections = [{'track_id': 1, 'class_id': 0, 'centroid': [50, 50], 'bbox': [10, 10, 90, 90]}]
        small_buffer.add_results(0, 0, detections)
        assert 0 in small_buffer.pending[0]['results']

    def test_add_results_nonexistent_frame_ignored(self, small_buffer):
        detections = [{'track_id': 1, 'class_id': 0}]
        small_buffer.add_results(99, 0, detections)
        assert 99 not in small_buffer.pending

    def test_id_namespacing(self, small_buffer, dummy_frame):
        small_buffer.add_frame(0, dummy_frame)
        detections = [{'track_id': 5, 'class_id': 2, 'centroid': [50, 50]}]

        small_buffer.add_results(0, 1, detections)  # backend_idx=1

        result = small_buffer.pending[0]['results'][1][0]
        assert result['track_id'] == 1_000_005  # 1 * 1_000_000 + 5
        assert result['class_id'] == 1002         # 1 * 1000 + 2


class TestMarkComplete:
    def test_mark_complete(self, small_buffer, dummy_frame):
        small_buffer.add_frame(0, dummy_frame)
        assert small_buffer.pending[0]['complete'] is False
        small_buffer.mark_complete(0)
        assert small_buffer.pending[0]['complete'] is True

    def test_mark_nonexistent(self, small_buffer):
        small_buffer.mark_complete(99)  # Should not raise


class TestMoveToBuffer:
    def test_sequential_move(self, small_buffer, dummy_frame):
        for i in range(3):
            small_buffer.add_frame(i, dummy_frame)
            small_buffer.mark_complete(i)

        small_buffer.try_move_to_buffer()
        assert len(small_buffer.buffer) == 3
        assert small_buffer.next_frame_to_play == 3

    def test_incomplete_blocks_move(self, small_buffer, dummy_frame):
        small_buffer.add_frame(0, dummy_frame)
        small_buffer.add_frame(1, dummy_frame)
        small_buffer.mark_complete(0)
        # Frame 1 NOT complete

        small_buffer.try_move_to_buffer()
        assert len(small_buffer.buffer) == 1  # Only frame 0 moved
        assert small_buffer.next_frame_to_play == 1

    def test_gap_blocks_move(self, small_buffer, dummy_frame):
        # Add frames 0 and 2 (gap at 1)
        small_buffer.add_frame(0, dummy_frame)
        small_buffer.add_frame(2, dummy_frame)
        small_buffer.mark_complete(0)
        small_buffer.mark_complete(2)

        small_buffer.try_move_to_buffer()
        assert len(small_buffer.buffer) == 1  # Only frame 0 — frame 1 missing
        assert small_buffer.next_frame_to_play == 1

    def test_buffer_maxlen_respected(self, dummy_frame):
        buf = SyncBuffer(num_backends=1, buffer_size=3)
        for i in range(5):
            buf.add_frame(i, dummy_frame)
            buf.mark_complete(i)

        buf.try_move_to_buffer()
        assert len(buf.buffer) == 3  # Capped at maxlen


class TestPlayback:
    def test_ready_for_playback(self, dummy_frame):
        buf = SyncBuffer(num_backends=1, buffer_size=3)
        for i in range(3):
            buf.add_frame(i, dummy_frame)
            buf.mark_complete(i)
        buf.try_move_to_buffer()
        assert buf.is_ready_for_playback() is True

    def test_not_ready_for_playback(self, small_buffer, dummy_frame):
        small_buffer.add_frame(0, dummy_frame)
        small_buffer.mark_complete(0)
        small_buffer.try_move_to_buffer()
        assert small_buffer.is_ready_for_playback() is False

    def test_get_next_frame(self, small_buffer, dummy_frame):
        small_buffer.add_frame(0, dummy_frame)
        small_buffer.mark_complete(0)
        small_buffer.try_move_to_buffer()

        data = small_buffer.get_next_frame()
        assert data is not None
        assert data['frame_id'] == 0

    def test_get_next_frame_empty(self, small_buffer):
        assert small_buffer.get_next_frame() is None

    def test_should_rebuffer(self, small_buffer, dummy_frame):
        small_buffer.add_frame(0, dummy_frame)
        small_buffer.mark_complete(0)
        small_buffer.try_move_to_buffer()
        assert small_buffer.should_rebuffer(threshold=5) is True


class TestGetStatus:
    def test_status_initial(self, small_buffer):
        status = small_buffer.get_status()
        assert status == {"buffer": 0, "pending": 0, "next": 0, "processed": 0}

    def test_status_with_data(self, small_buffer, dummy_frame):
        small_buffer.add_frame(0, dummy_frame)
        small_buffer.add_frame(1, dummy_frame)
        small_buffer.mark_complete(0)
        small_buffer.try_move_to_buffer()

        status = small_buffer.get_status()
        assert status["buffer"] == 1
        assert status["pending"] == 1
        assert status["next"] == 1
        assert status["processed"] == 1
