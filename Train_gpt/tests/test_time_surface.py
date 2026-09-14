import torch

from vsa_motionseg.data.time_surface import TimeSurfaceBuilder, events_window


def test_decay_and_polarity():
    events = torch.tensor(
        [
            [0.0, 10.0, 10.0, 1.0],
            [0.0, 20.0, 20.0, 0.0],
            [1.0, 10.0, 10.0, 1.0],
        ]
    )
    tsb = TimeSurfaceBuilder(40, 40, legacy_decay=0.8)
    pack = tsb.from_events(events, t_end=1.0)
    assert pack["surface"].shape == (2, 40, 40)
    assert pack["surface"][0, 10, 10] > pack["surface"][0, 0, 0]
    assert pack["active_mask"][10, 10]
    assert pack["event_count"][10, 10] >= 2


def test_events_window():
    ev = torch.tensor([[0.0, 0, 0, 0], [1.0, 0, 0, 0], [2.0, 0, 0, 0]])
    w = events_window(ev, 0.5, 1.5)
    assert w.shape[0] == 1
