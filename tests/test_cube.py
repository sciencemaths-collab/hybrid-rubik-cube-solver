from hybrid_rubik.cube import SOLVED, apply_move, scramble, astar


def test_move_inverse():
    s = apply_move(SOLVED, "R")
    assert apply_move(s, "R'") == SOLVED


def test_four_turns_identity():
    s = SOLVED
    for _ in range(4):
        s = apply_move(s, "U")
    assert s == SOLVED


def test_short_scramble_solves():
    s = scramble(["R", "U", "F"])
    r = astar(s, max_nodes=100000)
    assert r.solved
    out = s
    for m in r.moves:
        out = apply_move(out, m)
    assert out == SOLVED
