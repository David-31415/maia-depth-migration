"""mining.py — PGN mining for the Fork Around and Find Out experiments.

Everything the notebook mines from "lichess_pgns.pgn" in one shippable module:

1. FORK MINING — find clean royal forks by any piece.
     is_clean_knight_fork(board, move)          strictest: landing square unattacked
     is_standing_fork(board, move, piece_type)  fork that survives every recapture
     is_queen_kr_fork(board, move)              queen forks K + a LOOSE rook (the
                                                queen fork that wins material; K+Q
                                                queen "forks" are mostly trades)
     tag_any_standing_fork(board, move)         one predicate, all five piece types
     scan_pgn_for_forks(...)                    single-process scan
     scan_pgn_parallel(...)                     multi-process scan (bool predicates)
     scan_pgn_parallel_tagged(...)              multi-process scan (tag predicates)
     build_move_cache() / load_move_cache()     scan the whole corpus once -> move_cache.json

2. MOVE-QUALITY MINING — engine-graded move classes for ablation studies.
     classify_move(board, move, piece_type)     "quiet" / "attacking" / None (structural)
     mine_move_classes(pgn_path, piece_type, policy_fn, ...)
                                                good-quiet / good-attacking / bad-quiet /
                                                bad-attacking sets, graded by YOUR engine
                                                via policy_fn(board) -> [(uci, prob), ...]

The module is engine-agnostic: fork mining is pure python-chess, and move-quality
mining takes any `policy_fn` returning legal moves ordered best-first with
probabilities (e.g. `lambda b: eng.evaluate(b, 2700)["policy"]` for MAIA-3).

Build or refresh the fork cache from the command line:
    python mining.py

Why a module and not notebook cells: multiprocessing on macOS spawns fresh
interpreters for worker processes — they need to `import mining` to see these
functions by name, which a closure defined inside a notebook cell can't provide.

Scanning speed (two behavior-preserving speedups over a naive
`chess.pgn.read_game` + `mainline_moves()` loop; identical fork sets, verified):
1. `_ForkVisitor` reads each game in one pass via `chess.pgn.BaseVisitor`,
   skipping the Game/GameNode tree construction nothing here uses (~1.8x).
2. The parallel scans shard games across workers by byte offset (every game
   starts with a `[Event ` header line, so shard boundaries are a cheap text
   scan). SAN parsing is pure-Python and CPU-bound, so this is real wall-clock
   parallelism, not GIL-limited threading.
"""
import json
import os
import time

import chess
import chess.pgn
from concurrent.futures import ProcessPoolExecutor

DEFAULT_WORKERS = min(8, os.cpu_count() or 4)

PGN_PATH = os.path.join(os.path.dirname(__file__), "lichess_pgns.pgn")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "move_cache.json")
_ALL_GAMES = 10_000_000  # sentinel: scans stop at real EOF well before this

KING_QUEEN = {chess.KING, chess.QUEEN}
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


# ═══════════════════════════ 1. fork predicates ═══════════════════════════

def is_clean_knight_fork(board, move):
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.KNIGHT:
        return False
    if board.is_capture(move):
        return False
    mover = piece.color
    board.push(move)
    try:
        landing = move.to_square
        if board.is_attacked_by(not mover, landing):
            return False
        attacked = board.attacks(landing) & board.occupied_co[not mover]
        targets = sum(board.piece_at(sq).piece_type in KING_QUEEN for sq in attacked)
        return targets >= 2
    finally:
        board.pop()


def is_standing_fork(board, move, piece_type):
    """K+Q fork that STANDS: non-capture, non-promotion move after which the mover attacks
    both king and queen, and every enemy capture of the forker is either illegal (the king
    can't take a defended piece) or loses material (a higher-value piece takes a defended
    forker).

    Why not reuse the knight set's stricter 'landing square unattacked' condition? It is
    STRUCTURALLY impossible outside knights: a pawn attacking the king is diagonally adjacent
    to it, so the forked king always attacks the landing square back; and a bishop/rook
    attacking the queen shares her line, so the forked queen always sees the forker. Pawn and
    ray forks never stand by being untouchable -- only by being defended. (This also means the
    knight fork is the unique fork type that can be literally safe, which is presumably why
    it is the canonical one.)

    Caveat for QUEEN: attacking the enemy queen with yours only requires the landing square
    be defended, so many hits are forced queen TRADES rather than material wins -- expect a
    dilutead set (see the notebook's §19 queen note)."""
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != piece_type:
        return False
    if board.is_capture(move) or move.promotion is not None:
        return False
    mover = piece.color
    board.push(move)
    try:
        landing = move.to_square
        attacked = board.attacks(landing) & board.occupied_co[not mover]
        targets = sum(board.piece_at(sq).piece_type in KING_QUEEN for sq in attacked)
        if targets < 2:
            return False
        defended = bool(board.attackers(mover, landing))
        for sq in board.attackers(not mover, landing):
            attacker = board.piece_at(sq).piece_type
            if attacker == chess.KING:
                if not defended:
                    return False          # the king just takes the forker
            elif VALUE[attacker] < VALUE[piece_type] or not defended:
                return False              # a profitable recapture exists
        return True
    finally:
        board.pop()


def is_queen_kr_fork(board, move):
    """Queen fork of KING + ROOK that wins material: a non-capture queen move after
    which the queen checks the enemy king and attacks an enemy rook with NO defender,
    with the same landing-square safety rule as is_standing_fork.

    This is the queen's substitute for the K+Q sets: attacking the enemy queen with
    yours mostly forces a queen TRADE (see is_standing_fork's queen caveat), but a
    king + loose-rook fork wins the rook outright once the king steps out of check."""
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.QUEEN:
        return False
    if board.is_capture(move):
        return False
    mover = piece.color
    board.push(move)
    try:
        landing = move.to_square
        attacked = board.attacks(landing) & board.occupied_co[not mover]
        king_hit = any(board.piece_at(sq).piece_type == chess.KING for sq in attacked)
        loose_rook = any(board.piece_at(sq).piece_type == chess.ROOK
                         and not board.attackers(not mover, sq)
                         for sq in attacked)
        if not (king_hit and loose_rook):
            return False
        defended = bool(board.attackers(mover, landing))
        for sq in board.attackers(not mover, landing):
            attacker = board.piece_at(sq).piece_type
            if attacker == chess.KING:
                if not defended:
                    return False
            elif VALUE[attacker] < VALUE[chess.QUEEN] or not defended:
                return False
        return True
    finally:
        board.pop()


_TAG_BY_PIECE = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
                  chess.ROOK: "rook", chess.QUEEN: "queen"}


def tag_any_standing_fork(board, move):
    """Single predicate covering all five fork-capable piece types, so a full
    corpus scan only has to visit each move once instead of once per piece
    type. Returns the piece-type tag string, or None if `move` isn't a clean
    K+Q fork."""
    piece = board.piece_at(move.from_square)
    if piece is None:
        return None
    pt = piece.piece_type
    if pt == chess.KNIGHT:
        return "knight" if is_clean_knight_fork(board, move) else None
    if pt in (chess.PAWN, chess.BISHOP, chess.ROOK, chess.QUEEN):
        return _TAG_BY_PIECE[pt] if is_standing_fork(board, move, pt) else None
    return None


# ═══════════════════════════ 2. PGN scanning ═══════════════════════════

class _ForkVisitor(chess.pgn.BaseVisitor):
    """Single-pass visitor: fork_fn(board_before_move, move, *fork_args) for
    every move, skipping Game/GameNode tree construction entirely."""

    def __init__(self, fork_fn, fork_args):
        self.fork_fn = fork_fn
        self.fork_args = fork_args
        self.forks = []

    def visit_move(self, board, move):
        if self.fork_fn(board, move, *self.fork_args):
            self.forks.append((board.fen(), move))

    def result(self):
        return self.forks


class _TaggedForkVisitor(chess.pgn.BaseVisitor):
    """Like _ForkVisitor, but fork_fn returns a tag (truthy, non-bool) instead
    of a plain bool, and that tag is stored alongside each hit."""

    def __init__(self, fork_fn, fork_args):
        self.fork_fn = fork_fn
        self.fork_args = fork_args
        self.forks = []

    def visit_move(self, board, move):
        tag = self.fork_fn(board, move, *self.fork_args)
        if tag:
            self.forks.append((board.fen(), move, tag))

    def result(self):
        return self.forks


def scan_pgn_for_forks(pgn_path, max_games, fork_fn=is_clean_knight_fork, fork_args=(),
                        verbose=True, offset=0):
    """Scan `max_games` games starting at byte `offset` (default: file start).
    Returns a list of (fen, move) pairs -- single-process."""
    forks = []
    step = max(max_games // 10, 1)
    with open(pgn_path) as pgn:
        pgn.seek(offset)
        for game_count in range(max_games):
            if verbose and game_count % step == 0:
                print(f"scanning game {game_count:>6} ... ({len(forks)} forks so far)")
            found = chess.pgn.read_game(pgn, Visitor=lambda: _ForkVisitor(fork_fn, fork_args))
            if found is None:
                break
            forks.extend(found)
    return forks


def _game_start_offsets(pgn_path, n_games):
    """Byte offset of the start of each of the first n_games games (a pure
    text scan -- no chess parsing, so this is fast even for n_games in the
    hundreds of thousands)."""
    offsets = []
    with open(pgn_path, "rb") as f:
        pos = 0
        for line in f:
            if line.startswith(b"[Event "):
                offsets.append(pos)
                if len(offsets) == n_games:
                    break
            pos += len(line)
    return offsets


def _shard_jobs(pgn_path, max_games, fork_fn, fork_args, workers):
    offsets = _game_start_offsets(pgn_path, max_games)
    n = len(offsets)
    if n == 0:
        return []
    chunk = (n + workers - 1) // workers
    jobs = []
    for w in range(workers):
        start = w * chunk
        end = min(start + chunk, n)
        if start >= end:
            break
        jobs.append((pgn_path, offsets[start], end - start, fork_fn, fork_args))
    return jobs


def _run_sharded(worker, jobs):
    forks = []
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        for i, chunk_forks in enumerate(ex.map(worker, jobs)):
            forks.extend(chunk_forks)
            print(f"worker {i+1}/{len(jobs)} done: {len(chunk_forks)} forks "
                  f"({len(forks)} total so far)")
    return forks


def _scan_worker(args):
    pgn_path, offset, n_games, fork_fn, fork_args = args
    return scan_pgn_for_forks(pgn_path, n_games, fork_fn, fork_args, verbose=False, offset=offset)


def _tagged_scan_worker(args):
    pgn_path, offset, n_games, fork_fn, fork_args = args
    forks = []
    with open(pgn_path) as pgn:
        pgn.seek(offset)
        for _ in range(n_games):
            found = chess.pgn.read_game(pgn, Visitor=lambda: _TaggedForkVisitor(fork_fn, fork_args))
            if found is None:
                break
            forks.extend(found)
    return forks


def scan_pgn_parallel(pgn_path, max_games, fork_fn=is_clean_knight_fork, fork_args=(),
                       workers=DEFAULT_WORKERS):
    """Same contract as scan_pgn_for_forks, but shards the first `max_games`
    games across `workers` processes. Order of returned forks is not
    guaranteed to match the sequential scan (shards run concurrently), but the
    *set* of forks found is identical."""
    jobs = _shard_jobs(pgn_path, max_games, fork_fn, fork_args, workers)
    return _run_sharded(_scan_worker, jobs) if jobs else []


def scan_pgn_parallel_tagged(pgn_path, max_games, fork_fn=tag_any_standing_fork, fork_args=(),
                              workers=DEFAULT_WORKERS):
    """Same sharding scheme as scan_pgn_parallel, but for tag-returning
    predicates. Returns (fen, move, tag) triples."""
    jobs = _shard_jobs(pgn_path, max_games, fork_fn, fork_args, workers)
    return _run_sharded(_tagged_scan_worker, jobs) if jobs else []

# ═══════════════════════════ 3. fork cache ═══════════════════════════

def build_move_cache(pgn_path=PGN_PATH, cache_path=CACHE_PATH):
    """Scan the full corpus once for all five fork types and cache to JSON,
    so later analysis (depth-ladder sweeps, test sets, ...) never re-scans."""
    start = time.time()
    forks = scan_pgn_parallel_tagged(pgn_path, _ALL_GAMES)

    by_piece = {}
    for fen, move, tag in forks:
        by_piece.setdefault(tag, []).append({"fen": fen, "move": move.uci()})

    # Queen K+R forks use their own predicate (the K+Q "queen" set is mostly forced
    # trades), so they get a second scan and their own cache key. A move forking
    # K, Q, and a loose R lands in both queen sets -- that overlap is intended.
    kr = scan_pgn_parallel(pgn_path, _ALL_GAMES, fork_fn=is_queen_kr_fork)
    by_piece["queen_kr"] = [{"fen": fen, "move": move.uci()} for fen, move in kr]
    elapsed = time.time() - start

    payload = {
        "source": os.path.basename(pgn_path),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scan_seconds": round(elapsed, 1),
        "counts": {k: len(v) for k, v in by_piece.items()},
        "forks": by_piece,
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f)

    print(f"\nScanned in {elapsed:.1f}s. Counts: {payload['counts']}")
    print(f"Cached to {cache_path}")
    return payload


def load_move_cache(cache_path=CACHE_PATH):
    """Returns {"pawn": [{"fen":..., "move":...}, ...], "knight": [...], ...}.
    Raises FileNotFoundError with a hint if the cache hasn't been built yet."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"{cache_path} not found -- run `python mining.py` once to build it."
        )
    with open(cache_path) as f:
        payload = json.load(f)
    return payload["forks"]


# ═══════════════════════════ 4. move-quality mining ═══════════════════════════

def _enemy_attacked_squares(board, mover):
    """Set of enemy-occupied squares the mover currently attacks."""
    return {sq for sq in chess.SquareSet(board.occupied_co[not mover])
            if board.is_attacked_by(mover, sq)}


def _hangs_piece(board, mover, landing, piece_type):
    """True if the piece the mover just placed on `landing` can be won
    outright, using the same recapture test as is_standing_fork: the king
    takes it undefended, or a piece worth less than `piece_type` takes it."""
    defended = bool(board.attackers(mover, landing))
    for sq in board.attackers(not mover, landing):
        attacker = board.piece_at(sq).piece_type
        if attacker == chess.KING:
            if not defended:
                return True
        elif VALUE[attacker] < VALUE[piece_type] or not defended:
            return True
    return False


def _threatens(board, mover, sq, attacker_value):
    """True if the mover's attack on the enemy piece at `sq` demands a
    response: the piece is undefended, or it is worth more than the piece
    attacking it (so the capture wins material through the recapture)."""
    target = board.piece_at(sq).piece_type
    if target == chess.KING:
        return False
    return not board.attackers(not mover, sq) or VALUE[target] > attacker_value


def classify_move(board, move, piece_type):
    """Structural move class for ablation studies:
        "quiet"     -- creates no new attacks on enemy pieces
        "attacking" -- creates a new attack that genuinely demands a response
                       (target undefended, or worth more than the attacker)
        None        -- capture / promotion / check / hangs the piece /
                       creates only toothless attacks / ambiguous
    Board is left unchanged.

    Note the class is purely structural: "attacking" certifies that the
    threat is real, NOT that making it is a good idea. Whether a real-looking
    threat actually accomplishes anything is a positional judgment left to
    the engine grading in mine_move_classes -- which is what makes its
    bad-attacking set mean "showy but pointless" rather than "toothless"."""
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != piece_type:
        return None
    if board.is_capture(move) or move.promotion is not None:
        return None
    mover = piece.color
    before = _enemy_attacked_squares(board, mover)
    board.push(move)
    try:
        if board.is_check():
            return None
        landing = move.to_square
        if _hangs_piece(board, mover, landing, piece_type):
            return None
        new_hits = (board.attacks(landing) & board.occupied_co[not mover]) - before
        after = _enemy_attacked_squares(board, mover)
        if not new_hits and after <= before:
            return "quiet"
        if new_hits and any(_threatens(board, mover, sq, VALUE[piece_type])
                            for sq in new_hits):
            return "attacking"
        return None
    finally:
        board.pop()


def mine_move_classes(pgn_path, piece_type, policy_fn,
                      classes=("good-quiet", "good-attacking",
                               "bad-quiet", "bad-attacking"),
                      quota=50, max_games=1500, min_ply=16, ply_stride=3,
                      bad_min_rank=10, bad_max_prob=0.02, per_game_cap=2,
                      verbose=True):
    """Mine engine-graded move classes for one piece type.

    "good-X"  = the engine's TOP choice (rank 0) that classify_move calls X.
    "bad-X"   = a legal X move the engine ranks >= `bad_min_rank` with
                probability < `bad_max_prob` (robustly pointless, not merely
                second-best).

    The four defaults cross move quality with attacking-ness, which is what
    makes the pair identifiable: bad-quiet isolates "bad and aimless" while
    bad-attacking isolates "bad despite making a real threat" -- the showy
    move. Comparing those two is the whole point; mining only bad-quiet (the
    pre-2026-07 default) cannot separate the two hypotheses.

    policy_fn(board) must return the engine's legal moves best-first as
    [(uci, probability), ...] — e.g. for MAIA-3:
        policy_fn = lambda b: eng.evaluate(b, 2700)["policy"]
    (policy_fn is only called on positions that have at least one candidate
    move of the requested piece type, so mining stays cheap.)

    Returns {class_name: [(fen, uci), ...]} with up to `quota` items each.
    At most `per_game_cap` positions are taken per game so no single game
    dominates a class."""
    groups = {c: [] for c in classes}
    n_games = 0
    with open(pgn_path) as f:
        while any(len(v) < quota for v in groups.values()) and n_games < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            n_games += 1
            from_this_game = 0
            board = game.board()
            for ply, mv in enumerate(game.mainline_moves()):
                if ply >= min_ply and ply % ply_stride == 0 and from_this_game < per_game_cap:
                    cands = [(m, classify_move(board, m, piece_type)) for m in board.legal_moves]
                    cands = [(m, c) for m, c in cands if c]
                    if cands:
                        policy = policy_fn(board)
                        order = [u for u, _ in policy]
                        prob = dict(policy)
                        for m, c in cands:
                            u = m.uci()
                            rank = order.index(u) if u in order else None
                            if rank is None:
                                continue
                            name = None
                            if rank == 0:
                                name = f"good-{c}"
                            elif rank >= bad_min_rank and prob.get(u, 1) < bad_max_prob:
                                name = f"bad-{c}"
                            if name in groups and len(groups[name]) < quota:
                                groups[name].append((board.fen(), u))
                                from_this_game += 1
                                break
                board.push(mv)
    if verbose:
        print(f"mined over {n_games} games: { {k: len(v) for k, v in groups.items()} }")
    return groups


if __name__ == "__main__":
    build_move_cache()


_SEE_VALUE = {**VALUE, chess.KING: 100}


def exchange_loss(board, move):
    """Material the enemy nets by capturing the piece `move` just landed, taking the
    exchange on that square to the end with the cheapest attacker first (0 when every
    capture loses). No x-rays. The one-capture rule in is_standing_fork misses the case a
    pile of attackers wins against a single defender."""
    b = board.copy()
    b.push(move)
    sq = move.to_square

    def gain(b, side, on):
        for s in sorted(b.attackers(side, sq), key=lambda s: _SEE_VALUE[b.piece_at(s).piece_type]):
            a = b.piece_at(s).piece_type
            if a == chess.KING and b.attackers(not side, sq):
                continue
            c = b.copy()
            c.remove_piece_at(s)
            c.set_piece_at(sq, chess.Piece(a, side))
            return max(0, on - gain(c, not side, _SEE_VALUE[a]))
        return 0

    return gain(b, b.turn, _SEE_VALUE[board.piece_at(move.from_square).piece_type])
