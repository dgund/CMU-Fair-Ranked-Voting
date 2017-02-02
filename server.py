#!/usr/bin/env python2

from flask import Flask, send_from_directory, request

import json
from cStringIO import StringIO
import sys

from stv_compute import Ballot
from stv_compute import Election

app = Flask(__name__)

@app.route('/stv')
def stv():
    data = json.loads(request.args['data'])
    raw_ballots = data['ballots']
    _candidates = data['candidates']
    seats = data['seats']

    ballots = set()

    for ballot_number, raw_ballot in raw_ballots.items():
        ballot = Ballot()
        for rank, candidate in raw_ballot.items():
            if candidate and rank.isdigit():
                 ballot.set_candidate_with_rank(candidate.strip(), int(rank))
        ballots.add(ballot)

    election = Election()
    election.seats = seats
    election.ballots = ballots

    old_stdout = sys.stdout
    sys.stdout = output = StringIO()

    _winners, counter = election.compute_winners(verbose=True)

    sys.stdout = old_stdout

    output = {
        'winners': list(counter.winning_candidates),
        'losers': list(counter.losing_candidates),
        'votes': counter._votes_for_candidate_per_round,
        'quota': election.droop_quota(len(ballots), seats),
        'output': output.getvalue()
    }

    return json.dumps(output)

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    port = sys.argv[1] if len(sys.argv) > 1 else 8000
    app.run(port=port)
