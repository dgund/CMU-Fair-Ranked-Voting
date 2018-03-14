
import numpy as np
import pandas as pd
from typing import Iterable

import argparse
import logging
import string

# logging.basicConfig(level=logging.INFO)


def stv(raw_ballots, seats, eliminate_NC=True, tiebreak=None):
    # type: (Iterable, int, bool, str) -> set
    """
    :param raw_ballots: np.array, columns are preferences, rows are voters
        number of seats inferred from number of columns
    :param seats: number of seats
    :param eliminate_NC: whether it's possible to eliminate No Confidence
    :param tiebreak: alphanumeric to break ties
    :return: winning candidates

    >>> stv([['A']]*10, 1) == {'A'}
    True
    >>> stv([['A', 'B']]*5 + [['B', 'A']]*10, 1) == {'B'}
    True
    >>> stv([['A', 'C']]*5 + [['B']]*10 + [['C']]*7, 1) == {'C'}
    True
    >>> stv([['A']]*10, 2) == {'A'}
    True
    >>> stv([['A']]*7 + [['B', 'C']]*10 + [['C']]*7, 2) == {'B', 'C'}
    True
    >>> stv([['A', 'NC']]*10 + [['B', 'NC']]*16 +
    ...     [['C', 'NC']]*15 + [['D', 'A']]*7, 2) == {'A', 'B'}
    True

    Test No Confidence
    >>> stv([['A', 'NC']]*5 + [['B', 'NC']]*6 +
    ...     [['C', 'NC']]*4 + [['NC']]*5, 3) == {'B', 'NC'}
    True
    >>> stv([['A']]*10 + [['B']]*6 + [['NC']]*8, 3) == {'A', 'NC'}
    True
    >>> stv([['A', 'NC']]*6 + [['B', 'NC']]*5 + [['C', 'NC']]*1, 1) == {'A'}
    True
    >>> stv([['A', 'NC']]*6 + [['B', 'NC']]*5 + [['C', 'NC']]*1, 1, False
    ...     ) == {'NC'}
    True


    Test tie breaks
    >>> stv([['A']]*3 + [['B']]*3 + [['C', 'B']]*1 + [['D', 'A']]*1, 2
    ...     ) == {'A', 'B'}
    True
    >>> stv([['A']]*6 + [['B']]*3 + [['C']]*2 + [['D', 'C']]*1, 2) == {'A', 'B'}
    True
    >>> stv([['A']]*6 + [['B', 'C']]*3 + [['C']]*3, 2) == {'A', 'C'}
    True
    >>> stv([['A']]*6 + [['B']]*3 + [['C']]*3, 2) == {'A', 'C'}
    True

    Test large elections
    >>> scale = 100
    >>> stv(
    ...     [['tarsier', 'gorilla']]*scale*5 +
    ...     [['gorilla', 'tarsier', 'monkey']]*scale*28 +
    ...     [['monkey']]*scale*33 +
    ...     [['tiger']]*scale*21 +
    ...     [['lynx', 'tiger', 'tarsier', 'monkey', 'gorilla']]*scale*13,
    ... 3) == {'gorilla', 'monkey', 'tiger'}
    True
    >>> stv(
    ...     [['tarsier', 'silverback']]*scale*5 +
    ...     [['gorilla', 'silverback']]*scale*21 +
    ...     [['gorilla', 'tarsier', 'silverback']]*scale*11 +
    ...     [['silverback']]*scale*3 +
    ...     [['owl', 'turtle']]*scale*33 +
    ...     [['turtle']]*scale*1 +
    ...     [['snake', 'turtle']]*scale*1 +
    ...     [['tiger']]*scale*16 +
    ...     [['lynx', 'tiger']]*scale*4 +
    ...     [['jackalope']]*scale*2 +
    ...     [['buffalo', 'jackalope']]*scale*2 +
    ...     [['buffalo', 'jackalope', 'turtle']]*scale*1,
    ... 5) == {'gorilla', 'silverback', 'owl', 'turtle', 'tiger'}
    True
    >>> stv(
    ...     [['oranges']]*4 +
    ...     [['pears', 'oranges']]*2 +
    ...     [['chocolate', 'strawberries']]*8 +
    ...     [['chocolate', 'sweets']]*4 +
    ...     [['strawberries']]*1 +
    ...     [['sweets']]*1,
    ... 3) == {'chocolate', 'oranges', 'strawberries'}
    True
    >>> stv(
    ...     [['Bush']]*29127 +
    ...     [['Gore']]*29122 +
    ...     [['Nader', 'Bush']]*324 +
    ...     [['Nader', 'Gore']]*649,
    ... 1) == {'Gore'}
    True
    >>> stv(
    ...     [['G', 'F', 'H']]*14 +
    ...     [['J']]*12 +
    ...     [['F', 'G']]*11 +
    ...     [['A', 'B', 'C']]*11 +
    ...     [['D', 'E', 'A']]*8 +
    ...     [['D', 'E', 'A', 'B']]*8 +
    ...     [['D', 'E', 'C', 'F']]*8 +
    ...     [['E', 'D', 'F', 'G', 'H']]*8 +
    ...     [['E', 'D', 'G']]*8 +
    ...     [['D', 'E', 'NC']]*8 +
    ...     [['I', 'A', 'B', 'C']]*7 +
    ...     [['H', 'G']]*6 +
    ...     [['C', 'B', 'A']]*6 +
    ...     [['J', 'NC']]*6 +
    ...     [['B', 'A', 'C']]*3 +
    ...     [['I', 'A', 'C', 'B']]*3,
    ... 6) == {'D', 'E', 'G', 'A', 'F', 'J'}
    True
    """

    default_tiebreak = "0123456789abcdefghijklmnopqrstuvwxyz"
    tiebreak = tiebreak or default_tiebreak

    log = logging.getLogger("stv")
    # input can be any kind of iterable, including generator
    # pd.DataFrame looks like the most straightforward way to get a consistent
    # array of ballots of uniform size preserving order and filling the rest
    # with NaNs
    ballots = pd.DataFrame(raw_ballots).values
    candidates = [c for c in pd.unique(ballots.ravel())
                  if c and pd.notnull(c)]
    winners = set()

    # how much of each position was casted from each ballot
    votes = np.zeros(ballots.shape)
    # if candidate at valid[ballot, position] is valid
    valid = pd.notnull(ballots)
    stats = pd.Series()
    stats_history = []

    def least_preferred_candidate():
        """ Implementation of tiebreak rules """
        lpc = stats.sort_values()
        if not eliminate_NC and 'NC' in lpc:
            lpc = lpc.drop('NC')
        # it looks like bulk elimination is the same as sequential
        lpc = lpc[lpc == lpc.min()].keys()
        if len(lpc) == 1:
            return lpc[0]
        # backward tie break
        log.info("There is a tie for a candidate to be eliminated: %s", lpc)
        log.info("First, let's lok at prior rounds to find who scored less")
        for snapshot in reversed(stats_history[:-1]):
            shortlist = snapshot.reindex(lpc, fill_value=0)
            log.info("Counts: %s", dict(shortlist))
            lpc = shortlist[shortlist == shortlist.min()].keys()
            if len(lpc) == 1:
                log.info("Apparently %s is the least liked", lpc[0])
                return lpc[0]
            log.info("There is no clear anti winner, let's try one step back")

        log.info("Ok, backward tie break didn't work. let's look who has "
                  "the most next choice votes")
        # count, how many votes will each of remaining candidates have
        # if all others are eliminated
        scores = pd.Series(0, index=lpc)
        tb_valid = valid.copy()
        for candidate in lpc:
            tb_valid[ballots == candidate] = False
        for candidate in lpc:
            tb_valid[ballots == candidate] = True
            for i in range(len(ballots)):
                # almost a copy from vote count below
                rv = 1 - votes[i].dot(tb_valid[i])
                ao = np.where(tb_valid[i] * (votes[i] == 0))[0]
                if ao.size > 0 and rv > 0:
                    npc = ballots[i, ao.min()]
                    if npc in scores:
                        stats[npc] += rv
            tb_valid[ballots == candidate] = False
        log.info("Next scores: \n, %s", dict(scores))
        lpc = scores[scores == scores.min()].keys()
        if len(lpc) == 1:
            log.info("Apparently %s is the least liked", lpc[0])
            return lpc[0]

        log.info("Ok, they're all equal. Now going to use random tiebreak")
        lpc = sorted(lpc, key=lambda x: x.translate(string.maketrans(tiebreak, default_tiebreak)))
        return lpc[0]

    # Actual STV.
    for elections_round in range(len(ballots)):  # same as while True + counter
        # count votes
        for i in range(len(ballots)):
            # at each round, move remaining vote right
            remaining_vote = 1 - votes[i].dot(valid[i])
            # np.where returns #n-dimensions tuple. We've 1D, so [0]
            available_options = np.where(valid[i] * (votes[i] == 0))[0]
            if available_options.size > 0 and remaining_vote > 0:
                next_preference_idx = available_options.min()
                next_preference_candidate = ballots[i, next_preference_idx]
                votes[i, next_preference_idx] = remaining_vote
                if next_preference_candidate not in stats:
                    stats[next_preference_candidate] = 0.0
                stats[next_preference_candidate] += remaining_vote
        # for backward tie breaking
        stats_history.append(stats.copy())

        total = votes[valid][map(lambda x: x not in winners, ballots[valid])].sum()
        threshold = total * 1.0 / (seats - len(winners) + 1) + 1

        winners_before = len(winners)
        # check if there are winners. redistribute votes/end if necessary
        for candidate, score in stats.sort_values(ascending=False).items():
            if candidate in winners:
                continue
            if pd.isnull(score) or score < threshold:
                break
            winners.add(candidate)
            # redistribute excess, if any
            # don't worry about people checking for new winners -they'll be
            # calculated automatically on the next round
            votes[ballots == candidate] *= threshold / score
            stats[candidate] *= threshold / score

        log.info("Results after round %d:", elections_round)
        log.info("Ballots:\n%s", ballots)
        log.info("Valid choices:\n%s", valid)
        log.debug("Votes:\n%s", pd.DataFrame(
            np.concatenate((ballots, votes * valid), axis=1),
            columns=["ballot_%d" % i for i in range(ballots.shape[1])] +
                    ["vote_%d" % i for i in range(ballots.shape[1])]
            ))
        log.info("Stats:\n%s", stats.T)
        log.info("Threshold: %s", threshold)
        log.info("Winners: %s", winners)

        if len(winners) == seats or 'NC' in winners:
            return winners

        if len(winners) == winners_before:
            # eliminate least popular
            lp = least_preferred_candidate()
            if lp in winners:  # too many people abstained`
                log.info("Ran out of candidates, some seats remain vacant")
                return winners
            log.info("Eliminating %s as the least popular", lp)
            del(stats[lp])
            valid[ballots == lp] = False


if __name__ == "__main__":
    pass
