import pytest
import numpy as np
import copy

from lrl.data_stores import GeneralIterationData, EpisodeStatistics, DictWithHistory


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
    ws = EpisodeStatistics()
    for data in supply_ws_data:
        ws.add(*data)
    return ws


def test_WalkStatistics_add(supply_ws_data, ws_sample):
    # ws = EpisodeStatistics()
    # ws.add(5, [(0, 0), (2, 0), (10, 0)], True)
    # ws.add(3, [(0, 0), (5, 0), (10, 0)], False)
    # ws.add(12, [(1, 1), (3, 3)], True)
    ws = ws_sample

    # Independent data to compare to (uses same source as ws_sample)
    rewards, episodes, terminals = [list(data) for data in zip(*supply_ws_data)]
    # rewards = list(rewards)
    # episodes = list(episodes)
    # terminals = list(terminals)

    steps = [len(episode) for episode in episodes]
    statistics = [None for episode in episodes]

    assert ws.terminals == terminals
    assert ws.rewards == rewards
    assert ws.steps == steps
    assert ws._statistics == statistics
    assert ws.episodes[1] == episodes[1]


def test_EpisodeStatistics_compute(ws_sample):
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
            'episode_index': i,
            'terminal': terminals[i],
            'terminal_fraction': terminals[:i+1].sum() / (i+1),
        })

    # Test computing incrementally
    for i in range(len(rewards)):
        ws_incremental.compute(index=i)
        assert ws_incremental._statistics[i] == statistics[i]

    # Test computing statistics all at once
    for i in range(len(rewards)):
        ws_all_at_once.compute(index='all')
    assert ws_incremental._statistics == ws_all_at_once._statistics


def test_DictWithHistory():
    # Test Explicit timepointing
    dh = DictWithHistory()

    # Test with some floats.  Hundreds position is same as as key.  Ones position is same as timepoint
    n_tests = 5
    for k, x in enumerate(range(n_tests)):
        dh[k] = float(x * 100)

    # Test if everything has right value with timepoint == 0
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
        assert dh.get_value_at_timepoint(5, 0)

    # Test if to_dict writes out the given timepoint's values as a dictionary
    # Initial timepoint data
    original = {0: 10.0,
                1: 100.0,
                2: 200.0,
                3: 300.0,
                4: 400.0,
                }
    assert dh.to_dict(timepoint=0) == pytest.approx(original)

    # Most recent data (default)
    recent = {0: 2.0,
              1: 100.0,
              2: 200.0,
              3: 302.0,
              4: 400.0,
              5: 501.0,
              }
    assert dh.to_dict() == pytest.approx(recent)

    # Test implicit timepointing
    dh = DictWithHistory(timepoint_mode='implicit')

    # Test with some floats.  Hundreds position is same as as key.  Ones position is same as timepoint
    n_tests = 5
    for k, x in enumerate(range(n_tests)):
        dh[k] = float(x * 100 + x)

    # Test if everything has right value with first assignments
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
def supply_DictWithHistory_simple():
    """
    Return a simple DictWithHistory Factory (use this rather than direct instance for multiple DH in same test)
    Returns:
        A factory that generates simple DictWithHistory instances
    """
    class Factory:
        @staticmethod
        def get():
            dh = DictWithHistory(timepoint_mode='explicit')
            dh['a'] = 1.0
            dh['b'] = 2.0
            dh['c'] = 3.0
            dh.increment_timepoint()
            dh['b'] = 2.0  # Shouldn't update anything
            dh['c'] = 3.1  # Should update timepoint
            return dh
    return Factory()


def test_DictWithHistory_update(supply_DictWithHistory_simple):
    dh1 = supply_DictWithHistory_simple.get()
    dh2 = supply_DictWithHistory_simple.get()
    dh3 = supply_DictWithHistory_simple.get()

    d = {'a': 11.1, 'b': 12.1, 'c': 13.1}
    d2 = {'a': 11.2, 'b': 12.2, 'c': 13.2}

    # Update on timepoint=explicit should add everything from a dict to current timepoint
    for k, v in d.items():
        dh1[k] = v

    dh2.update(d)
    assert dh1.to_dict() == dh2.to_dict()
    assert dh1._data == dh2._data
    assert dh1.current_timepoint == dh2.current_timepoint

    # If timepoint_mode == 'implicit', all entries should still go to same timepoint (but timepoint will increment
    # after update)
    dh3.timepoint_mode = 'implicit'
    dh3.update(d)
    assert dh1.to_dict() == dh3.to_dict()
    assert dh1._data == dh3._data
    assert dh1.current_timepoint + 1 == dh3.current_timepoint

    # And adding more data will be like I incremented the timepoint manually
    dh1.increment_timepoint()
    for k, v in d2.items():
        dh1[k] = v

    dh3.update(d2)
    assert dh1.to_dict() == dh3.to_dict()
    assert dh1._data == dh3._data
    assert dh1.current_timepoint + 1 == dh3.current_timepoint


def test_DictWithHistory_in_dict_differences(supply_DictWithHistory_simple):
    """
    Test DictWithHistory in dict_differences, which should interpret DHS as a dictionary of its most recent data
    """
    # Test Explicit timepointing
    dh1 = supply_DictWithHistory_simple.get()
    dh2 = supply_DictWithHistory_simple.get()
    dh3 = supply_DictWithHistory_simple.get()

    # for dh in [dh1, dh2, dh3]:
    #     dh['a'] = 1.0
    #     dh['b'] = 2.0
    #     dh['c'] = 3.0
    #     dh.increment_timepoint()
    #     dh['b'] = 2.0  # Shouldn't update anything
    #     dh['c'] = 3.1  # Should update timepoint

    assert dh1.to_dict() == dh2.to_dict()

    dh2.increment_timepoint()
    dh2['c'] = 3.2
    assert dh1.to_dict() != dh2.to_dict()

    dh3['c'] = 1000.0
    assert dh1.to_dict() != dh3.to_dict()


def test_DictWithHistory_tuple():
    dh1 = DictWithHistory()
    dh2 = DictWithHistory()
    dh3 = DictWithHistory()
    dh4 = DictWithHistory()

    for dh in [dh1, dh2, dh3, dh4]:
        dh['a'] = (0, 1, 2.5)
        dh['b'] = (10, 11, 12.5)

    assert dh1.to_dict(timepoint=0) == dh2.to_dict(timepoint=0)

    # This update should be noticed as a duplicate and no change made to dh2
    dh2.increment_timepoint()
    dh2['a'] = (0, 1, 2.5)

    assert dh1.current_timepoint != dh2.current_timepoint  # Check timepoints are different just in case
    assert dh1.to_dict(timepoint=0) == dh2.to_dict(timepoint=0)

    # This update should be too small of a change to notice
    dh2['a'] = (0, 1, 2.5 + 1e-08)
    assert dh1.to_dict(timepoint=0) == dh2.to_dict(timepoint=0)
    assert dh1._data == dh2._data

    # This should actually be a change.  to_dict(timepoint=-0) wont show it, but to_dict(timepoint=-1) will
    dh2['a'] = (0, 1, 3.5)
    assert dh1.to_dict(timepoint=0) == dh2.to_dict(timepoint=0)
    assert dh1.to_dict(timepoint=-1) != dh2.to_dict(timepoint=-1)

    # This change is a different structure of data entirely - definitely a change
    dh2['a'] = [10000, 20000]
    assert dh1.to_dict() != dh2.to_dict()

    # Non-numeric shouldn't break everything, but will definitely show as a change!
    dh3['a'] = "not_a_number"
    assert dh1.to_dict() != dh3.to_dict()

    dh4['a'] = "not_a_number"
    dh4.increment_timepoint()
    # This should not change things, either in the outward dict or in the backend
    dh4['a'] = "not_a_number"
    assert dh3.to_dict() == dh4.to_dict()
    assert dh3._data == dh4._data


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