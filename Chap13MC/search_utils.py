""" search_utils module

This module implements functions that return the index of an input in an iterable using bisect

"""

from bisect import bisect_left, bisect_right


def index(a, x):
    """Locate the leftmost value exactly equal to x

        Args:
            a  (iterable): iterable to search
            x: lookup_value

        Returns:
            int: index in iterable

        Raises:
           ValueError
    """
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


def find_lt(a, x):
    """Find rightmost value less than x

        Args:
            a  (iterable): iterable to search
            x: lookup_value

        Returns:
            type: value

        Raises:
           ValueError
    """
    i = bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError


def find_le(a, x):
    """Find rightmost value less than or equal to x

        Args:
            a  (iterable): iterable to search
            x: lookup_value

        Returns:
           type: value

        Raises:
           ValueError
    """
    i = bisect_right(a, x)
    if i:
        return a[i-1]
    raise ValueError


def find_gt(a, x):
    """  Find leftmost value greater than x

        Args:
            a  (iterable): iterable to search
            x: lookup_value

        Returns:
            type: value

        Raises:
           ValueError
    """
    i = bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError


def find_ge(a, x):
    """Find leftmost item greater than or equal to x

        Args:
            a  (iterable): iterable to search
            x: lookup_value

        Returns:
            type: value

        Raises:
           ValueError
    """
    i = bisect_left(a, x)
    if i != len(a):
        return a[i]
    raise ValueError


def index_lt(a, x):
    """Find leftmost index less than x

        Args:
            a  (iterable): iterable to search
            x: lookup_value

        Returns:
            int: index in iterable
    """
    i = bisect_left(a, x)
    return i - 1


def index_gt(a, x):
    """Find rightmost index greater than x

        Args:
            a  (iterable): iterable to search
            x: lookup_value

        Returns:
            int: index in iterable
    """
    i = bisect_right(a, x)
    return i - 1
