import pytest

from lrl.utils.misc import params_to_name


def test_params_to_name():
    # Test different types
    assert params_to_name({'e': 10.2, 'd': 5}) == 'd_5__e_10.2'
    assert params_to_name({'e': (1, 2), 'd': 5}) == 'd_5__e_(1, 2)'

    # Test sep
    assert params_to_name({'e': 10.2, 'd': 5}, sep='&') == 'd&5&&e&10.2'

    # Test truncation
    assert params_to_name({'elong': 10.2, 'dlonger': 5}) == 'dlon_5__elon_10.2'
    assert params_to_name({'elong': 10.2, 'dlonger': 5}, n_chars=6) == 'dlonge_5__elong_10.2'

    # Test nested mappings
    assert params_to_name({'e': {'d': 5, 'f': 10}}) == 'e_{d_5__f_10}'
    assert params_to_name({'e': {'d': 5, 'flong': 10}}) == 'e_{d_5__flon_10}'

    # Test first_fields
    assert params_to_name({'e': {'d': 5, 'flong': 10, 'e': 60}, 'a': 0}) == 'a_0__e_{d_5__e_60__flon_10}'
    assert params_to_name({'e': {'d': 5, 'flong': 10, 'e': 60}, 'a': 0}, first_fields=['e', 'not_here']) \
           == 'e_{e_60__d_5__flon_10}__a_0'

    # Test key_remap
    assert params_to_name({'e': 10.2, 'd': 5}, key_remap={'e': 'othr'}) == 'd_5__othr_10.2'
