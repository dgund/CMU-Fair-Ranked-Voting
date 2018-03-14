#!/usr/bin/env python3

"""Unit tests for election.py."""

from __future__ import print_function
import unittest

from election import Ballot, Candidate, Election, NoConfidence

__author__ = "Devin Gund"
__copyright__ = "Copyright 2017, Carnegie Mellon University Undergraduate Student Senate"
__credits__ = ["Sushain Cherivirala"]
__license__ = "GPLv3"
__status__ = "Production"

CANDIDATE_IDS = {
    'A': ('dgund', 'Devin Gund'),
    'B': ('gwashington', 'George Washington'),
    'C': ('jadams', 'John Adams'),
    'D': ('tjefferson', 'Thomas Jefferson'),
    'E': ('jmadison', 'James Madison'),
    'F': ('jmonroe', 'James Monroe'),
    'G': ('jqadams', 'John Quincy Adams'),
    'H': ('ajackson', 'Andrew Jackson'),
    'I': ('mvburen', 'Martin Van Buren'),
    'J': ('wharrison', 'William Harrison'),
    'K': ('jtyler', 'John Tyler'),
    'L': ('jpolk', 'James Polk'),
    'M': ('ztaylor', 'Zachary Taylor'),
    'N': ('mfillmore', 'Millard Fillmore'),
    'O': ('fpierce', 'Franklin Pierce'),
    'P': ('jbuchanan', 'James Buchanan'),
    'Q': ('alincoln', 'Abraham Lincoln'),
    'R': ('ajohnson', 'Andrew Johnson'),
    'S': ('ugrant', 'Ulysses Grant'),
    'T': ('rhayes', 'Rutherford Hayes'),
    'U': ('jgarfield', 'James Garfield'),
    'V': ('carthur', 'Chester Arthur'),
    'W': ('gcleveland', 'Grover Cleveland'),
    'X': ('bharrison', 'Benjamin Harrison'),
    'Y': ('wmckinley', 'William McKinley'),
    'Z': ('troosevelt', 'Theodore Roosevelt'),
}


def candidates_for_ids(candidate_ids):
    """Returns an ordered list of Candidates for an ordered list of ids.

    This is helpful for tests where candidates on ballots can be defined as
    short strings (i.e. 'A', 'B', 'NC') in the test cases, which can then be
    converted to proper Candidates.

    :param candidate_ids: List of strings representing candidate ids.

    :return List of Candidates corresponding to the given ids.
    """

    return [Candidate(CANDIDATE_IDS[candidate_id])
            if candidate_id != "NC" else NoConfidence()
            for candidate_id in candidate_ids]


def ballots_for_candidates(candidates, count):
    """Returns a list of Ballots for the given Candidates and count.

    :param candidates: List of Candidates.
    :param count: Integer value of the number of Ballots to create.

    :return List of Ballots for the given Candidates, with the given count.
    """
    return [Ballot(candidates) for _ in range(count)]


def ballots_for_ids(candidate_ids, count):
    """Returns a list of Ballots for the given ids and count.

    :param candidate_ids: List of strings representing candidate ids.
    :param count: Integer value of the number of Ballots to create.

    :return List of Ballots for the given ids, with the given count.
    """
    return ballots_for_candidates(candidates_for_ids(candidate_ids), count)


class TestSmallElections(unittest.TestCase):

    def test_1_candidate_1_seat(self):
        """Tests a 1 candidate election for 1 seat.

        Expected winners: A

        Round 0
            Ballots:
                10 * [A]
            Votes:
                A: 10
            Threshold: (10) / (1+1) + 1 = 6
            Result: A is elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['A']))
        seats = 1
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'
        ballots = (
            ballots_for_ids(['A'], 10))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_2_candidates_1_seat(self):
        """Tests a 2 candidate election for 1 seat.

        Expected winners: B

        Round 0
            Ballots:
                 5 * [A, B]
                10 * [B, A]
            Votes:
                A: 5
                B: 10
            Threshold: (15) / (1+1) + 1 = 8.5
            Result: B is elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['B']))
        seats = 1
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A', 'B'], 5) +
            ballots_for_ids(['B', 'A'], 10))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_3_candidates_1_seat(self):
        """Tests a 3 candidate election for 1 seat.

        Expected winners: C

        Round 0
            Ballots:
                 5 * [A, C]
                10 * [B]
                 7 * [C]
            Votes:
                A: 7
                B: 10
                C: 5
            Threshold: (7+10+5) / (1+1) + 1 = 12
            Result: A is eliminated

        Round 1
            Ballots:
                10 * [B]
                12 * [C]
            Votes:
                B: 10
                C: 12
            Threshold: (7+10+5) / (1+1) + 1 = 12
            Result: C is elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['C']))
        seats = 1
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A', 'C'], 5) +
            ballots_for_ids(['B'], 10) +
            ballots_for_ids(['C'], 7))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_1_candidate_2_seats(self):
        """Tests a 1 candidate election for 2 seats.

        Expected winners: A

        Round 0
            Ballots:
                 10 * [A]
            Votes:
                A: 10
            Threshold: (10) / (2+1) + 1 = 4.333
            Result: A is elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['A']))
        seats = 2
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A'], 10))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_3_candidates_2_seats(self):
        """Tests a 3 candidate election for 2 seats.

        Expected winners: B, C

        Round 0
            Ballots:
                 7 * [A]
                10 * [B, C]
                 7 * [C]
            Votes:
                A: 7
                B: 10
                C: 7
            Threshold: (7+10+7) / (2+1) + 1 = 9
            Result: B is elected, with a surplus of 1
            Each ballot is redistributed with value * (1/10) = value * .1

        Round 1
            Ballots:
                7 * [A]
                10 * [C] (with value .1)
                7 * [C]
            Votes:
                A: 7
                C: 8
            Threshold: (7+8) / (1+1) + 1 = 8.5
            Result: A is eliminated

        Round 2
            Ballots:
                10 * [C] (with value .1)
                7 * [C]
            Votes:
                C: 8
            Threshold: (8) / (1+1) + 1 = 5
            Result: C is elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['B', 'C']))
        seats = 2
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A'], 7) +
            ballots_for_ids(['B', 'C'], 10) +
            ballots_for_ids(['C'], 7))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_4_candidates_2_seats(self):
        """Tests a 4 candidate election for 2 seats.

        Expected winners: A, B

        Round 0
            Ballots:
                10 * [A, NC]
                16 * [B, NC]
                15 * [C, NC]
                 7 * [D, A]
            Votes:
                A: 10
                B: 16
                C: 15
                D: 7
            Threshold: (10+16+15+7) / (2+1) + 1 = 17
            Result: D is eliminated

        Round 1
            Ballots:
                10 * [A, NC]
                16 * [B, NC]
                15 * [C, NC]
                 7 * [A]
            Votes:
                A: 17
                B: 16
                C: 15
            Threshold: (10+16+15+7) / (2+1) + 1 = 17
            Result: A is elected

        Round 2
            Ballots:
                16 * [B, NC]
                15 * [C, NC]
            Votes:
                B: 16
                C: 15
            Threshold: (16+15) / (1+1) + 1 = 16.5
            Result: C is eliminated

        Round 3
            Ballots:
                16 * [B, NC]
                15 * [NC]
            Votes:
                B: 16
                NC: 15
            Threshold: (16+15) / (1+1) + 1 = 16.5
            Result: NC is eliminated

        Round 4
            Ballots:
                16 [B, NC]
            Votes:
                B: 16
            Threshold: (16+15) / (1+1) + 1 = 16.5
            Result: B is elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['A', 'B']))
        seats = 2
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A', 'NC'], 10) +
            ballots_for_ids(['B', 'NC'], 16) +
            ballots_for_ids(['C', 'NC'], 15) +
            ballots_for_ids(['D', 'A'], 7))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)


class TestNoConfidence(unittest.TestCase):

    def test_nc_halt_early(self):
        """ Tests a 3 candidate election for 3 seats.

        Once No Confidence is elected, the election should end.

        Expected winners: B, NC

        Round 0
            Ballots:
                5 * [A, NC]
                6 * [B, NC]
                4 * [C, NC]
                5 * [NC]
            Votes:
                A: 5
                B: 6
                C: 4
               NC: 5
            Threshold: (6+5+4+5) / (3+1) + 1 = 6
            Result: B is elected

        Round 1
            Ballots:
                5 * [A, NC]
                4 * [C, NC]
                5 * [NC]
            Votes:
                A: 5
                C: 4
               NC: 5
            Threshold: (5+4+5) / (2+1) + 1 = 5.666
            Result: C is eliminated

        Round 2
            Ballots:
                5 * [A, NC]
                9 * [NC]
            Votes:
                A: 5
               NC: 9
            Threshold: (5+9) / (2+1) + 1 = 5.666
            Result: NC is elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['B', 'NC']))
        seats = 3
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A', 'NC'], 5) +
            ballots_for_ids(['B', 'NC'], 6) +
            ballots_for_ids(['C', 'NC'], 4) +
            ballots_for_ids(['NC'], 5))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_nc_halt_end(self):
        """Tests a 2 candidate election for 3 seats.

        When empty seats are filled, only candidates with votes exceeding that
        of No Confidence may be elected.

        Expected winners: A, NC

        Round 0
            Ballots:
               10 * [A]
                6 * [B]
                8 * [NC]
            Votes:
                A: 10
                B: 6
               NC: 8
            Threshold: (10+6+8) / (3+1) + 1 = 7
            Result: A is elected
            As the 3 candidates (including No Confidence) can fill the 3 vacant
            seats, the candidates are slated to fill the seats. However, B's
            votes fall below that of No Confidence, so B is not elected.
        """
        # Setup
        expected_winners = set(candidates_for_ids(['A', 'NC']))
        seats = 3
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A'], 10) +
            ballots_for_ids(['B'], 6) +
            ballots_for_ids(['NC'], 8))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_election_with_nc_elimination(self):
        """Tests a 3 candidate election for 1 seat.

        All of the ballots list a single candidate followed by No Confidence.

        Expected winners: A

        Round 0
            Ballots:
                6 * [A, NC]
                5 * [B, NC]
                1 * [C, NC]
            Votes:
                A: 6
                B: 5
                C: 1
            Threshold: (6+5+1) / (1+1) + 1 = 7
            Result: C is eliminated

        Round 1
            Ballots:
                6 * [A, NC]
                5 * [B, NC]
                1 * [NC]

            Votes:
                A: 6
                B: 5
                NC: 1
            Threshold: (6+5+1) / (1+1) + 1 = 7
            Result: NC is eliminated

        Round 2
            Ballots:
                6 * [A]
                5 * [B]

            Votes:
                A: 6
                B: 5
            Threshold: (6+5) / (1+1) + 1 = 6.5
            Result: B is eliminated

        Round 3
            Ballots:
                6 * [A]

            Votes:
                A: 6
            Threshold: (6) / (1+1) + 1 = 4
            Result: A is elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['A']))
        seats = 1
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A', 'NC'], 6) +
            ballots_for_ids(['B', 'NC'], 5) +
            ballots_for_ids(['C', 'NC'], 1))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_election_without_nc_elimination(self):
        """Tests a 3 candidate election for 1 seat.

        All of the ballots list a single candidate followed by No Confidence.
        No Confidence is cannot be eliminated for having the fewest votes.

        Expected winners: NC

        Round 0
            Ballots:
                6 * [A, NC]
                5 * [B, NC]
                1 * [C, NC]
            Votes:
                A: 6
                B: 5
                C: 1
            Threshold: (6+5+1) / (1+1) + 1 = 7
            Result: C is eliminated

        Round 1
            Ballots:
                6 * [A, NC]
                5 * [B, NC]
                1 * [NC]
            Votes:
                A: 6
                B: 5
                NC: 1
            Threshold: (6+5+1) / (1+1) + 1 = 7
            Result: B is eliminated (because NC cannot be)

        Round 2
            Ballots:
                6 * [A, NC]
                6 * [NC]
            Votes:
                A: 6
                NC: 6
            Threshold: (6+6) / (1+1) + 1 = 7
            Result: A is eliminated (because NC cannot be)

        Round 3
            Ballots:
                12 * [NC]
            Votes:
                NC: 12
            Threshold: (12) / (1+1) + 1 = 7
            Result: NC is elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['NC']))
        seats = 1
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A', 'NC'], 6) +
            ballots_for_ids(['B', 'NC'], 5) +
            ballots_for_ids(['C', 'NC'], 1))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric,
                            can_eliminate_no_confidence=False)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)


class TestTiebreaks(unittest.TestCase):

    def test_bulk_elimination(self):
        """Tests a 4 candidate election for 2 seats.

        In the event of a tie to eliminate a candidate, of the tied candidates'
        combined vote total is lass than that of the next highest candidate,
        eliminate all of the tied candidates.

        Expected winners: A, B

        Round 0
            Ballots:
                3 * [A]
                3 * [B]
                1 * [C, B]
                1 * [D, A]
            Votes:
                A: 3
                B: 3
                C: 1
                D: 1
            Threshold: (3+3+1+1) / (2+1) + 1 = 3.666
            Result: C and D are eliminated
            C and D are tied for fewest votes, but their combined vote total (2)
            is less than the next highest candidate (3), so they may be
            eliminated in bulk.

        Round 0
            Ballots:
                4 * [A]
                4 * [B]
            Votes:
                A: 4
                B: 4
            Threshold: (4+4) / (2+1) + 1 = 3.666
            Result: A and B are elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['A', 'B']))
        seats = 2
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A'], 3) +
            ballots_for_ids(['B'], 3) +
            ballots_for_ids(['C', 'B'], 1) +
            ballots_for_ids(['D', 'A'], 1))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_backward_tiebreak(self):
        """Tests a 4 candidate election for 2 seats.

        In the event of a tie to eliminate a candidate, eliminate the candidate
        with the fewest votes in the previous round. Repeat for all previous
        rounds if necessary.

        Expected winners: A, B

        Round 0
            Ballots:
                6 * [A]
                3 * [B]
                2 * [C]
                1 * [D, C]
            Votes:
                A: 6
                B: 3
                C: 3
                D: 1
            Threshold: (6+3+2+1) / (2+1) + 1 = 5
            Result: A is elected

        Round 1
            Ballots:
                3 * [B]
                2 * [C]
                1 * [D, C]
            Votes:
                B: 3
                C: 3
                D: 1
            Threshold: (3+2+1) / (1+1) + 1 = 4
            Result: D is eliminated

        Round 2
            Ballots:
                3 * [B]
                3 * [C]
            Votes:
                B: 3
                C: 3
            Threshold: (3+3) / (1+1) + 1 = 4
            Result: C is eliminated
            B and C are tied for fewest votes, but C has fewer votes in the
            previous round, so C is eliminated.

        Round 3
            Ballots:
                3 * [B]
            Votes:
                B: 3
            Threshold: (3) / (1+1) + 1 = 2.5
            Result: B is elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['A', 'B']))
        seats = 2
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A'], 6) +
            ballots_for_ids(['B'], 3) +
            ballots_for_ids(['C'], 2) +
            ballots_for_ids(['D', 'C'], 1))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_forward_tiebreak(self):
        """Tests a 3 candidate election for 2 seats.

        In the event of a tie to eliminate a candidate, and no backward
        tiebreak, eliminate the candidate with the fewest next-choice votes.
        Repeat for all subsequent ranks as necessary.

        Expected winners: A, C

        Round 0
            Ballots:
                6 * [A]
                3 * [B, C]
                3 * [C]
            Votes:
                A: 6
                B: 3
                C: 3
            Threshold: (6+3+3) / (2+1) + 1 = 5
            Result: A is elected

        Round 1
            Ballots:
                3 * [B, C]
                3 * [C]
            Votes:
                B: 3
                C: 3
            Threshold: (3+2+1) / (1+1) + 1 = 4
            Result: B is eliminated
            B and C are tied for fewest votes in this and previous rounds.
            B has fewer next-rank votes, so B is eliminated.

        Round 2
            Ballots:
                6 * [C]
            Votes:
                C: 6
            Threshold: (6) / (1+1) + 1 = 4
            Result: C is elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['A', 'C']))
        seats = 2
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A'], 6) +
            ballots_for_ids(['B', 'C'], 3) +
            ballots_for_ids(['C'], 3))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_random_tiebreak(self):
        """Tests a 3 candidate election for 2 seats.

        In the event of a tie to eliminate a candidate, and no backward
        tiebreak or forward tiebreak, eliminate a random candidate based on uid.

        Expected winners: A, C

        Round 0
            Ballots:
                6 * [A]
                3 * [B]
                3 * [C]
            Votes:
                A: 6
                B: 3
                C: 3
            Threshold: (6+3+3) / (2+1) + 1 = 5
            Result: A is elected

        Round 1
            Ballots:
                3 * [B]
                3 * [C]
            Votes:
                B: 3
                C: 3
            Threshold: (3+3) / (1+1) + 1 = 4
            Result: B is elected
            B and C are tied for fewest votes in this and all previous rounds,
            as well as future ranks. B's uid ('gwashington') is ordered before
            C's uid ('jadams') in the tiebreak_alphanumeric, so B is eliminated.

        Round 2
            Ballots:
                3 * [C]
            Votes:
                C: 3
            Threshold: (3) / (1+1) + 1 = 2.5
            Result: C is elected
        """
        # Setup
        expected_winners = set(candidates_for_ids(['A', 'C']))
        seats = 2
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['A'], 6) +
            ballots_for_ids(['B'], 3) +
            ballots_for_ids(['C'], 3))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)


class TestLargeElections(unittest.TestCase):

    def test_cgp_grey_animal_kingdom(self):
        """Tests CGP Grey example STV election 'Politics in the Animal Kingdom'

        Link: https://www.youtube.com/watch?v=l8XOZJkozfI

        Expected winners: Gorilla, Monkey, Tiger
        """
        # Setup
        tarsier = Candidate('tarsier', name='Tarsier')
        gorilla = Candidate('gorilla', name='Gorilla')
        monkey = Candidate('monkey', name='Monkey')
        tiger = Candidate('tiger', name='Tiger')
        lynx = Candidate('lynx', name='Lynx')

        expected_winners = {gorilla, monkey, tiger}
        seats = 3
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        num_ballots = 10000
        ballots = (
            ballots_for_candidates([tarsier, gorilla],
                                   int(.05 * num_ballots)) +
            ballots_for_candidates([gorilla, tarsier, monkey],
                                   int(.28 * num_ballots)) +
            ballots_for_candidates([monkey],
                                   int(.33 * num_ballots)) +
            ballots_for_candidates([tiger],
                                   int(.21 * num_ballots)) +
            ballots_for_candidates([lynx, tiger, tarsier, monkey, gorilla],
                                   int(.13 * num_ballots)))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_cgp_grey_stv_election_walkthrough(self):
        """Tests CGP Grey example STV election 'Extra: STV Election Walkthrough'

        Link: https://www.youtube.com/watch?v=Ac9070OIMUg

        Expected winners: Gorilla, Silverback, Owl, Turtle, Tiger
        """
        # Setup
        tarsier = Candidate('tarsier', name='Tarsier')
        gorilla = Candidate('gorilla', name='Gorilla')
        silverback = Candidate('silverback', name='Silverback')
        owl = Candidate('owl', name='Owl')
        turtle = Candidate('turtle', name='Turtle')
        snake = Candidate('snake', name='Snake')
        tiger = Candidate('tiger', name='Tiger')
        lynx = Candidate('lynx', name='Lynx')
        jackalope = Candidate('jackalope', name='Jackalope')
        buffalo = Candidate('buffalo', name='Buffalo')

        expected_winners = {gorilla, silverback, owl, turtle, tiger}
        seats = 5
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        num_ballots = 10000
        ballots = (
            ballots_for_candidates([tarsier, silverback],
                                   int(.05 * num_ballots)) +
            ballots_for_candidates([gorilla, silverback],
                                   int(.21 * num_ballots)) +
            ballots_for_candidates([gorilla, tarsier, silverback],
                                   int(.11 * num_ballots)) +
            ballots_for_candidates([silverback],
                                   int(.03 * num_ballots)) +
            ballots_for_candidates([owl, turtle],
                                   int(.33 * num_ballots)) +
            ballots_for_candidates([turtle],
                                   int(.01 * num_ballots)) +
            ballots_for_candidates([snake, turtle],
                                   int(.01 * num_ballots)) +
            ballots_for_candidates([tiger],
                                   int(.16 * num_ballots)) +
            ballots_for_candidates([lynx, tiger],
                                   int(.04 * num_ballots)) +
            ballots_for_candidates([jackalope],
                                   int(.02 * num_ballots)) +
            ballots_for_candidates([buffalo, jackalope],
                                   int(.02 * num_ballots)) +
            ballots_for_candidates([buffalo, jackalope, turtle],
                                   int(.01 * num_ballots)))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_wikipedia_food_selection(self):
        """Tests Wikipedia's example STV election, using choices of food.

        Link: https://en.wikipedia.org/wiki/Single_transferable_vote#Example

        Expected winners: Chocolate, Oranges, Strawberries
        """
        # Setup
        chocolate = Candidate('chocolate', name='Chocolate')
        oranges = Candidate('oranges', name='Oranges')
        pears = Candidate('pears', name='Pears')
        strawberries = Candidate('strawberries', name='Strawberries')
        sweets = Candidate('sweets', name='Sweets')

        expected_winners = {chocolate, oranges, strawberries}
        seats = 3
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_candidates([oranges], 4) +
            ballots_for_candidates([pears, oranges], 2) +
            ballots_for_candidates([chocolate, strawberries], 8) +
            ballots_for_candidates([chocolate, sweets], 4) +
            ballots_for_candidates([strawberries], 1) +
            ballots_for_candidates([sweets], 1))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_florida_2000_presidential(self):
        """Tests the 2000 Florida presidential election, scaled 1%.

        An example of a election between two major candidates and a third-party.
        For the sake of ranking votes, this test assumes that Nader supporters
        prefer Gore to Bush by a 2:1 margin, which is not necessarily true.

        Expected winners: Gore

        Round 0
            Ballots:
              29127 * [Bush]
              29122 * [Gore]
                324 * [Nader, Bush]
                649 * [Nader, Gore]
            Votes:
                Bush: 29127
                Gore: 29122
                Nader:  974
            Threshold: (29127+29122+974) / (1+1) + 1 = 29612.5
            Result: Nader is eliminated

        Round 1
            Ballots:
                29452 * [Bush]
                29972 * [Gore]
            Votes:
                Bush: 29452
                Gore: 29972
            Threshold: (29452+29972) / (1+1) + 1 = 29612.5
            Result: Gore is elected
        """
        # Setup
        bush = Candidate('gbush', name='George Bush')
        gore = Candidate('agore', name='Al Gore')
        nader = Candidate('rnader', name='Ralph Nader')
        expected_winners = {gore}
        seats = 1
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_candidates([bush], 29127) +
            ballots_for_candidates([gore], 29122) +
            ballots_for_candidates([nader, bush], 324) +
            ballots_for_candidates([nader, gore], 649))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)

    def test_10_candidates_6_seats(self):
        """Tests a 10 candidate election for 6 seats

        Expected winners: D, E, G, A, F, J

        Candidates
        Party 1: A, B, C
        Party 2: D, E
        Party 3: F, G, H
        Party 4: I
        Party 5: J

        Context
        20 Party 1 voters
        11 * [A, B, C]
         6 * [C, B, A]
         3 * [B, A, C]
        48 Party 2 voters (othewise split between Party 1 and 3)
         8 * [D, E, A]
         8 * [D, E, A, B]
         8 * [D, E, C, F]
         8 * [D, E, NC]
         8 * [E, D, F, G, H]
         8 * [E, D, G]
        31 Party 3 voters
        14 * [G, F, H]
        11 * [F, G]
         6 * [H, G]
        10 Party 4 voters (otherwise support Party 1)
         7 * [I, A, B, C]
         3 * [I, A, C, B]
        18 Party 5 voters
        12 * [J]
         6 * [J, NC]

        Round 0
            Ballots:
                14 * [G, F, H]
                12 * [J]
                11 * [F, G]
                11 * [A, B, C]
                 8 * [D, E, A]
                 8 * [D, E, A, B]
                 8 * [D, E, C, F]
                 8 * [E, D, F, G, H]
                 8 * [E, D, G]
                 8 * [D, E, NC]
                 7 * [I, A, B, C]
                 6 * [H, G]
                 6 * [C, B, A]
                 6 * [J, NC]
                 3 * [B, A, C]
                 3 * [I, A, C, B]
            Votes:
                A: 11
                B: 3
                C: 6
                D: 32
                E: 16
                F: 11
                G: 14
                H: 6
                I: 10
                J: 18
            Threshold: (11+3+6+32+16+11+14+6+10+18) / (6+1) + 1 = 19.14285
            Result: D is elected with surplus 12.857
            Ballots redistributed with value * (12.857/32) = value * 0.401

         Round 1
            Ballots:
                14 * [G, F, H]
                12 * [J]
                11 * [F, G]
                11 * [A, B, C]
                 8 * [E, A] (worth 0.401)
                 8 * [E, A, B] (worth 0.401)
                 8 * [E, C, F] (worth 0.401)
                 8 * [E, F, G, H]
                 8 * [E, G]
                 8 * [E, NC] (worth 0.401)
                 7 * [I, A, B, C]
                 6 * [H, G]
                 6 * [C, B, A]
                 6 * [J, NC]
                 3 * [B, A, C]
                 3 * [I, A, C, B]
            Votes:
                A: 11
                B: 3
                C: 6
                E: 28.832
                F: 11
                G: 14
                H: 6
                I: 10
                J: 18
            Threshold: (11+3+6+28.832+11+14+6+10+18) / (5+1) + 1 = 18.972
            Result: E is elected with surplus 10.56
            Ballots redistributed with value * (10.56 / 28.832) = value * 0.366

         Round 2
            Ballots:
                14 * [G, F, H]
                12 * [J]
                11 * [F, G]
                11 * [A, B, C]
                 8 * [A] (worth 0.1467)
                 8 * [A, B] (worth 0.1467)
                 8 * [C, F] (worth 0.1467)
                 8 * [F, G, H] (with 0.366)
                 8 * [G] (worth 0.366)
                 8 * [NC] (worth 0.1467)
                 7 * [I, A, B, C]
                 6 * [H, G]
                 6 * [C, B, A]
                 6 * [J, NC]
                 3 * [B, A, C]
                 3 * [I, A, C, B]
            Votes:
                A: 13.347
                B: 3
                C: 7.174
                F: 13.928
                G: 16.928
                H: 6
                I: 10
                J: 18
                NC: 1.174
            Threshold: (13.347+3+7.174+13.928+16.928+6+10+18+1.174) / (4+1) + 1
                        = 18.91
            Result: NC is eliminated

         Round 3
            Ballots:
                14 * [G, F, H]
                18 * [J]
                11 * [F, G]
                11 * [A, B, C]
                 8 * [A] (worth 0.1467)
                 8 * [A, B] (worth 0.1467)
                 8 * [C, F] (worth 0.1467)
                 8 * [F, G, H] (with 0.366)
                 8 * [G] (worth 0.366)
                 7 * [I, A, B, C]
                 6 * [H, G]
                 6 * [C, B, A]
                 3 * [B, A, C]
                 3 * [I, A, C, B]
            Votes:
                A: 13.347
                B: 3
                C: 7.174
                F: 13.928
                G: 16.928
                H: 6
                I: 10
                J: 18
            Threshold: (13.347+3+7.174+13.928+16.928+6+10+18) / (4+1) + 1
                        = 18.675
            Result: B is eliminated

         Round 4
            Ballots:
                14 * [G, F, H]
                18 * [J]
                11 * [F, G]
                14 * [A, C]
                16 * [A] (worth 0.1467)
                 8 * [C, F] (worth 0.1467)
                 8 * [F, G, H] (with 0.366)
                 8 * [G] (worth 0.366)
                10 * [I, A, C]
                 6 * [H, G]
                 6 * [C, A]
            Votes:
                A: 16.347
                C: 7.174
                F: 13.928
                G: 16.928
                H: 6
                I: 10
                J: 18
            Threshold: (16.347+7.174+13.928+16.928+6+10+18) / (4+1) + 1 = 18.675
            Result: H is eliminated

         Round 5
            Ballots:
                14 * [G, F]
                18 * [J]
                11 * [F, G]
                14 * [A, C]
                16 * [A] (worth 0.1467)
                 8 * [C, F] (worth 0.1467)
                 8 * [F, G] (with 0.366)
                 8 * [G] (worth 0.366)
                10 * [I, A, C]
                 6 * [G]
                 6 * [C, A]
            Votes:
                A: 16.347
                C: 7.174
                F: 13.928
                G: 22.928
                H: 6
                I: 10
                J: 18
            Threshold: (16.347+7.174+13.928+22.928+6+10+18) / (4+1) + 1 = 19.875
            Result: G is elected with surplus 3.053
            Ballots redistributed with value * (3.053 / 19.875) = value * 0.1536

         Round 6
            Ballots:
                14 * [F] (worth 0.1536)
                18 * [J]
                11 * [F]
                14 * [A, C]
                16 * [A] (worth 0.1467)
                 8 * [C, F] (worth 0.1467)
                 8 * [F] (with 0.366)
                10 * [I, A, C]
                 6 * [C, A]
            Votes:
                A: 16.347
                C: 7.174
                F: 16.078
                I: 10
                J: 18
            Threshold: (16.347+7.174+16.078+10+18) / (3+1) + 1 = 17.9
            Result: C is eliminated

         Round 7
            Ballots:
                14 * [F] (worth 0.1536)
                18 * [J]
                11 * [F]
                20 * [A]
                16 * [A] (worth 0.1467)
                 8 * [F] (worth 0.1467)
                 8 * [F] (with 0.366)
                10 * [I, A]
            Votes:
                A: 22.347
                F: 17.252
                I: 10
                J: 18
            Threshold: (22.347+17.252+10+18) / (3+1) + 1 = 17.9
            Result: A is elected with surplus 4.447
            Ballots redistributed with value * (4.447 / 22.347) = value * 0.199

         Round 8
            Ballots:
                14 * [F] (worth 0.1536)
                18 * [J]
                11 * [F]
                 8 * [F] (worth 0.1467)
                 8 * [F] (with 0.366)
                10 * [I]
            Votes:
                F: 17.252
                I: 10
                J: 18
            Threshold: (17.252+10+18) / (2+1) + 1 = 16.084
            Result: F is elected with surplus 1.168 and
                    J is elected with surplus 1.916
        """
        # Setup
        expected_winners = set(candidates_for_ids(
                                ['D', 'E', 'G', 'A', 'F', 'J']))
        seats = 6
        tiebreak_alphanumeric = 'abcdefghijklmnopqrstuvwxyz'

        ballots = (
            ballots_for_ids(['G', 'F', 'H'], 14) +
            ballots_for_ids(['J'], 12) +
            ballots_for_ids(['F', 'G'], 11) +
            ballots_for_ids(['A', 'B', 'C'], 11) +
            ballots_for_ids(['D', 'E', 'A'], 8) +
            ballots_for_ids(['D', 'E', 'A', 'B'], 8) +
            ballots_for_ids(['D', 'E', 'C', 'F'], 8) +
            ballots_for_ids(['E', 'D', 'F', 'G', 'H'], 8) +
            ballots_for_ids(['E', 'D', 'G'], 8) +
            ballots_for_ids(['D', 'E', 'NC'], 8) +
            ballots_for_ids(['I', 'A', 'B', 'C'], 7) +
            ballots_for_ids(['H', 'G'], 6) +
            ballots_for_ids(['C', 'B', 'A'], 6) +
            ballots_for_ids(['J', 'NC'], 6) +
            ballots_for_ids(['B', 'A', 'C'], 3) +
            ballots_for_ids(['I', 'A', 'C', 'B'], 3))

        # Test
        election = Election(seats=seats,
                            ballots=ballots,
                            random_alphanumeric=tiebreak_alphanumeric)
        results = election.compute_results()
        self.assertEqual(expected_winners, results.candidates_elected)


if __name__ == '__main__':
    unittest.main()
