import pytest
import numpy as np
import copy

from lrl.data_stores import GeneralIterationData, WalkStatistics, DictWithHistory


@pytest.fixture
def supply_ws_data():
    """
    Supplies a list of walkstatistics.add inputs for building a test walk statistic object

    Returns:
        list
    """
    return [(5, [(0, 0), (2, 0), (10, 0)], True),
            (3, [(0, 0), (5, 0), (10, 0)], False),
            (12, [(1, 1), (3, 3)], True), ]


@pytest.fixture
def ws_sample(supply_ws_data):
    ws = WalkStatistics()
    for data in supply_ws_data:
        ws.add(*data)
    return ws


def test_WalkStatistics_add(supply_ws_data, ws_sample):
    # ws = WalkStatistics()
    # ws.add(5, [(0, 0), (2, 0), (10, 0)], True)
    # ws.add(3, [(0, 0), (5, 0), (10, 0)], False)
    # ws.add(12, [(1, 1), (3, 3)], True)
    ws = ws_sample

    # Independent data to compare to (uses same source as ws_sample)
    rewards, walks, terminals = [list(data) for data in zip(*supply_ws_data)]
    # rewards = list(rewards)
    # walks = list(walks)
    # terminals = list(terminals)

    steps = [len(walk) for walk in walks]
    statistics = [None for walk in walks]

    assert ws.terminals == terminals
    assert ws.rewards == rewards
    assert ws.steps == steps
    assert ws.statistics == statistics
    assert ws.walks[1] == walks[1]


def test_WalkStatistics_compute(ws_sample):
    ws = ws_sample
    ws_incremental = copy.deepcopy(ws)
    ws_all_at_once = copy.deepcopy(ws)
    ws_to_dataframe = copy.deepcopy(ws)

    rewards = np.array(ws.rewards)
    steps = np.array(ws.steps)
    terminals = np.array(ws.terminals)

    statistics = []
    for i in range(len(rewards)):
        statistics.append({
            'reward': rewards[i],
            'reward_mean': rewards[:i+1].mean(),
            'reward_median': np.median(rewards[:i+1]),
            'reward_std': rewards[:i+1].std(),
            'reward_max': rewards[:i+1].max(),
            'reward_min': rewards[:i+1].min(),
            'steps': steps[i],
            'steps_mean': steps[:i+1].mean(),
            'steps_median': np.median(steps[:i+1]),
            'steps_std': steps[:i+1].std(),
            'steps_max': steps[:i+1].max(),
            'steps_min': steps[:i+1].min(),
            'walk_index': i,
            'terminal': terminals[i],
            'terminal_fraction': terminals[:i+1].sum() / (i+1),
        })

    # Test computing incrementally
    for i in range(len(rewards)):
        ws_incremental.compute(index=i)
        assert ws_incremental.statistics[i] == statistics[i]

    # Test computing statistics all at once
    for i in range(len(rewards)):
        ws_all_at_once.compute(index='all')
    assert ws_incremental.statistics == ws_all_at_once.statistics


def test_DictWithHistory():
    # Test Explicit timepointing
    dh = DictWithHistory()

    # Test with some floats.  Hundreds position is same as as key.  Ones position is same as timepoint
    n_tests = 5
    for k, x in enumerate(range(n_tests)):
        dh[k] = float(x * 100)

    # Test if everything has right value with timepoint == 0
    print(dh._data)
    for i in range(n_tests):
        # Both direct access and using getter should be the same (although getter returns only value)
        assert dh._data[i] == [(0, pytest.approx(float(i * 100)))]
        assert dh[i] == pytest.approx(float(i * 100))

    # And if I change data, it changes the current timepoint
    dh[0] = 10.0
    assert dh._data[0] == [(0, pytest.approx(10.0))]
    assert dh[0] == pytest.approx(10.0)

    # Increment timepoint and add some new data
    dh.increment_timepoint()
    dh[3] = 301.0
    dh[3] = 311.0
    dh[5] = 501.0  # dh[5] initialized first here

    dh.increment_timepoint()
    dh[3] = 302.0
    dh[0] = 2.0
    # This should result in no change, because the value in here is already 100.0 (so timepoint 2 should not exist for key 0 when we test below)
    dh[1] = 100.0

    print(dh._data)

    assert dh._data[0] == [(0, pytest.approx(10.0)), (2, pytest.approx(2.0))]
    assert dh._data[1] == [(0, pytest.approx(100.0))]
    assert dh._data[3] == [(0, pytest.approx(300.0)), (1, pytest.approx(311.0)), (2, pytest.approx(302.0))]
    assert dh._data[5] == [(1, pytest.approx(501.0))]

    assert dh.get_value_at_timepoint(2, 0) == pytest.approx(200.0)
    assert dh.get_value_at_timepoint(5, 1) == pytest.approx(501.0)

    # Should raise KeyError (no key by this name)
    with pytest.raises(KeyError):
        assert dh.get_value_at_timepoint(10, 0)

    # Should raise IndexError (timepoint does not exist)
    with pytest.raises(KeyError):
        print(f"printing: {dh.get_value_at_timepoint(5, 0)}")
        assert dh.get_value_at_timepoint(1, 2)

    # Test if as_dict writes out the given timepoint's values as a dictionary
    # Initial timepoint data
    original = {0: 10.0,
                1: 100.0,
                2: 200.0,
                3: 300.0,
                4: 400.0,
                }
    assert dh.as_dict(timepoint=0) == pytest.approx(original)

    # Most recent data (default)
    recent = {0: 2.0,
              1: 100.0,
              2: 200.0,
              3: 302.0,
              4: 400.0,
              5: 501.0,
              }
    assert dh.as_dict() == pytest.approx(recent)

    # Test implicit timepointing
    dh = DictWithHistory(timepoint_mode='implicit')

    # Test with some floats.  Hundreds position is same as as key.  Ones position is same as timepoint
    n_tests = 5
    for k, x in enumerate(range(n_tests)):
        dh[k] = float(x * 100 + x)

    # Test if everything has right value with first assignments
    print(dh._data)
    for i in range(n_tests):
        # Both direct access and using getter should be the same (although getter returns only value)
        assert dh._data[i] == [(i, pytest.approx(float(i * 100 + i)))]
        assert dh[i] == pytest.approx(float(i * 100 + i))
    assert dh.current_timepoint == n_tests

    # And if I change data, it increments the timepoint the current timepoint
    dh[0] = 10.0
    assert dh._data[0] == [(0, pytest.approx(0.0)), (5, pytest.approx(10.0))]
    assert dh[0] == pytest.approx(10.0)


@pytest.fixture
def supply_generalIterationData():
    columns = ['iteration', 'time', 'delta_max']
    n_data = 5
    data = [{'iteration': i,
             'time': 10.0 + i,
             'delta_max': 100.0 + i}
            for i in range(n_data)]
    return columns, data


def test_GeneralIterationData(supply_generalIterationData):
    columns, data = supply_generalIterationData
    gid = GeneralIterationData(columns=columns)
    for i in range(len(data)):
        gid.add(data[i])

    # Test basic ingestion
    for i in range(len(data)):
        for column in columns:
            assert gid.get(i)[column] == pytest.approx(data[i][column])

    # Test using a subset of data just for a second opinion
    gid2 = GeneralIterationData()
    gid2.add(data[1])

    assert gid2.get(0)[columns[2]] == pytest.approx(data[1][columns[2]])