"""
Random Galaxies — Strong Divide & Conquer + Greedy AI
======================================================

TRUE D&C AI STRATEGY (computer_move):
--------------------------------------
DIVIDE:
  1. Find all missing solution edges (candidate moves).
  2. Build a "galaxy dependency graph" — nodes = galaxy rectangles,
     edges = which candidate borders separate which pairs of galaxies.
  3. Use BFS to split candidate edges into INDEPENDENT SUBPROBLEMS:
     each connected cluster of galaxies is solved separately.
     Clusters with no shared borders cannot affect each other => true D&C split.

CONQUER (recursive):
  4. For each sub-cluster, recursively apply the same D&C split after
     hypothetically placing the best local edge (look-ahead).
  5. Within each sub-cluster, rank every candidate edge by a GREEDY score:
       - TIER 1 (score 1000+): edge that COMPLETES a galaxy region right now
                                (instant bonus turn — highest priority)
       - TIER 2 (score 500+):  edge that sets up a completion on next move
                                (1-move lookahead)
       - TIER 3 (BFS score):   edge that most strongly separates two regions
                                (disconnects => N²+10, else BFS detour length)

COMBINE:
  6. Collect the best move from every sub-cluster.
  7. Merge-sort all sub-cluster winners by their score (descending).
  8. Play the globally best move across all sub-clusters.

DAA REQUIREMENTS:
  ✅ Divide & Conquer  — recursive board splitting into independent sub-clusters
  ✅ Greedy            — 3-tier scoring (complete > setup > separate)
  ✅ Merge Sort        — custom implementation, no built-in sort
  ✅ BFS components    — connected component detection
  ✅ BFS shortest path — edge impact scoring

GAME FEATURES:
  - Player edges = BLUE, Computer edges = RED
  - Completed galaxies highlighted green
  - Score tracking + bonus turns (completing a galaxy = extra turn)
  - Win condition: most completed galaxies when board is full
"""

import tkinter as tk
from tkinter import messagebox
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import random


# ============================================================
# MERGE SORT  (custom — no built-in sort anywhere in file)
# ============================================================

def merge_sort(arr: List, key=lambda x: x, reverse: bool = False) -> List:
    n = len(arr)
    if n <= 1:
        return arr[:]
    mid = n // 2
    left  = merge_sort(arr[:mid],  key=key, reverse=reverse)
    right = merge_sort(arr[mid:], key=key, reverse=reverse)
    merged, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = key(left[i]), key(right[j])
        if (a >= b) if reverse else (a <= b):
            merged.append(left[i]);  i += 1
        else:
            merged.append(right[j]); j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


# ============================================================
# BFS HELPERS
# ============================================================

def bfs_shortest_path(adj: Dict[int, List[int]], start: int, goal: int) -> Optional[int]:
    """BFS shortest path. Returns hop-count or None if unreachable."""
    if start == goal:
        return 0
    q = deque([start])
    dist = {start: 0}
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                if v == goal:
                    return dist[v]
                q.append(v)
    return None


def bfs_components(adj: Dict[int, List[int]], total_nodes: int) -> List[List[int]]:
    """BFS connected components over node IDs 0..total_nodes-1."""
    seen, comps = set(), []
    for s in range(total_nodes):
        if s in seen:
            continue
        q = deque([s])
        seen.add(s)
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(comp)
    return comps


def bfs_reachable(adj: Dict[int, List[int]], start: int) -> Set[int]:
    """Return all nodes reachable from start."""
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


# ============================================================
# PUZZLE GENERATOR  (rectangle tiling — always succeeds)
# ============================================================

Rect = Tuple[int, int, int, int]   # (x, y, w, h) in cell units


class GalaxiesPuzzle:
    def __init__(self, n: int = 7, rng: Optional[random.Random] = None):
        self.N = n
        self.rng = rng or random.Random()
        self.rects:          List[Rect]                      = []
        self.owner:          List[int]                       = [-1] * (n * n)
        self.dots:           List[Tuple[float, float]]       = []
        self.solution_edges: Set[Tuple[str, int, int]]       = set()

    def cell_id(self, x: int, y: int) -> int:
        return y * self.N + x

    def generate(self, target_rects: Optional[int] = None) -> None:
        n = self.N
        if target_rects is None:
            target_rects = self.rng.randint(9, 14)

        rects: List[Rect] = [(0, 0, n, n)]
        tries = 0
        while len(rects) < target_rects and tries < 5000:
            tries += 1
            candidates = [r for r in rects if r[2] >= 2 or r[3] >= 2]
            if not candidates:
                break
            r = self.rng.choice(candidates)
            rects.remove(r)
            x, y, w, h = r
            if w >= 2 and h >= 2:
                vertical = (w >= h and self.rng.random() < 0.65) or (self.rng.random() < 0.35)
            elif w >= 2:
                vertical = True
            else:
                vertical = False
            if vertical:
                k = self.rng.randint(1, w - 1)
                rects += [(x, y, k, h), (x + k, y, w - k, h)]
            else:
                k = self.rng.randint(1, h - 1)
                rects += [(x, y, w, k), (x, y + k, w, h - k)]

        self.rects = rects
        self.owner = [-1] * (n * n)
        for idx, (x, y, w, h) in enumerate(self.rects):
            for yy in range(y, y + h):
                for xx in range(x, x + w):
                    self.owner[self.cell_id(xx, yy)] = idx

        self.dots = [(x + w / 2.0, y + h / 2.0) for (x, y, w, h) in self.rects]
        self.solution_edges = self._compute_solution_edges()

    def _compute_solution_edges(self) -> Set[Tuple[str, int, int]]:
        n = self.N
        edges: Set[Tuple[str, int, int]] = set()
        for x in range(n):
            edges.add(('h', x, 0)); edges.add(('h', x, n))
        for y in range(n):
            edges.add(('v', 0, y)); edges.add(('v', n, y))
        for y in range(n):
            for x in range(n):
                o = self.owner[self.cell_id(x, y)]
                if x + 1 < n and self.owner[self.cell_id(x + 1, y)] != o:
                    edges.add(('v', x + 1, y))
                if y + 1 < n and self.owner[self.cell_id(x, y + 1)] != o:
                    edges.add(('h', x, y + 1))
        return edges


# ============================================================
# REGION COMPLETION HELPER
# ============================================================

def galaxy_is_complete(rect: Rect, edges: Set[Tuple[str, int, int]]) -> bool:
    """True when all 4 sides of the rectangle are in `edges`."""
    rx, ry, rw, rh = rect
    for x in range(rx, rx + rw):
        if ('h', x, ry) not in edges or ('h', x, ry + rh) not in edges:
            return False
    for y in range(ry, ry + rh):
        if ('v', rx, y) not in edges or ('v', rx + rw, y) not in edges:
            return False
    return True


def count_completed(puzzle: GalaxiesPuzzle, edges: Set[Tuple[str, int, int]]) -> int:
    return sum(1 for r in puzzle.rects if galaxy_is_complete(r, edges))


# ============================================================
# *** DIVIDE & CONQUER + GREEDY AI ***
# ============================================================

class DivideConquerAI:
    """
    TRUE D&C AI for Galaxies.

    Architecture
    ============
    divide()  — split missing edges into independent galaxy-cluster subproblems
    conquer() — recursively solve each sub-cluster with greedy scoring + lookahead
    combine() — merge-sort sub-cluster winners, return globally best move

    Greedy Scoring Tiers (within each sub-cluster)
    -----------------------------------------------
    TIER 1 (1000 + bonus): edge that immediately COMPLETES a galaxy (play now!)
    TIER 2 (500  + bonus): edge where placing it leaves an adjacent galaxy with
                           only 1 missing edge (setup for bonus turn next move)
    TIER 3 (BFS  score):   edge importance by BFS disconnection / detour length
                           — disconnects two regions  => N²+10
                           — long detour needed        => detour length
    """

    def __init__(self, game: 'GalaxiesGame'):
        self.game    = game
        self.puzzle  = game.puzzle
        self.N       = game.N

    # ----------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ----------------------------------------------------------

    def best_move(self) -> Optional[Tuple[str, int, int]]:
        """
        Full D&C pipeline:
          1. DIVIDE  — cluster missing edges into independent subproblems
          2. CONQUER — find best move inside each cluster (recursive)
          3. COMBINE — merge-sort all winners, return global best
        """
        missing = list(self.game.solution - (self.game.edges - self.game.fixed))
        if not missing:
            return None

        # ── DIVIDE ──────────────────────────────────────────────
        clusters = self._divide(missing, self.game.edges)

        if not clusters:
            return None

        # ── CONQUER ─────────────────────────────────────────────
        # For each cluster, recurse to find best local move + score
        sub_results: List[Tuple[int, Tuple[str, int, int]]] = []
        for cluster_edges in clusters:
            score, move = self._conquer(cluster_edges, self.game.edges, depth=0)
            if move is not None:
                sub_results.append((score, move))

        if not sub_results:
            return None

        # ── COMBINE ─────────────────────────────────────────────
        # Merge-sort sub-results descending by score, pick global best
        sorted_results = merge_sort(sub_results, key=lambda p: p[0], reverse=True)
        return sorted_results[0][1]

    # ----------------------------------------------------------
    # DIVIDE — split into independent sub-clusters
    # ----------------------------------------------------------

    def _divide(
        self,
        missing_edges: List[Tuple[str, int, int]],
        current_edges: Set[Tuple[str, int, int]]
    ) -> List[List[Tuple[str, int, int]]]:
        """
        Build a galaxy-adjacency graph:
          - Each galaxy rectangle is a node
          - Two galaxy nodes are connected if they share a missing border edge
        Then BFS on this galaxy-graph to find connected components.
        Each component is an INDEPENDENT subproblem (true D&C division).

        Why independent? Placing an edge inside cluster A can never affect
        whether galaxies in cluster B get completed — they share no borders.
        """
        # Map each missing edge -> the pair of galaxy indices it separates
        edge_to_galaxies: Dict[Tuple[str, int, int], Tuple[int, int]] = {}
        for e in missing_edges:
            gi = self._galaxies_separated_by(e)
            if gi is not None:
                edge_to_galaxies[e] = gi

        # Build galaxy adjacency graph (galaxy_idx -> set of adjacent galaxy_idx)
        galaxy_adj: Dict[int, Set[int]] = defaultdict(set)
        for (ga, gb) in edge_to_galaxies.values():
            galaxy_adj[ga].add(gb)
            galaxy_adj[gb].add(ga)

        # BFS over galaxy graph to find independent galaxy clusters
        all_galaxy_ids = set(galaxy_adj.keys())
        seen_galaxies: Set[int] = set()
        galaxy_clusters: List[Set[int]] = []

        for start in all_galaxy_ids:
            if start in seen_galaxies:
                continue
            cluster_galaxies: Set[int] = set()
            q = deque([start])
            seen_galaxies.add(start)
            while q:
                g = q.popleft()
                cluster_galaxies.add(g)
                for nb in galaxy_adj[g]:
                    if nb not in seen_galaxies:
                        seen_galaxies.add(nb)
                        q.append(nb)
            galaxy_clusters.append(cluster_galaxies)

        # Map clusters of galaxies -> clusters of edges
        edge_clusters: List[List[Tuple[str, int, int]]] = []
        for gc in galaxy_clusters:
            cluster_edges = [
                e for e, (ga, gb) in edge_to_galaxies.items()
                if ga in gc or gb in gc
            ]
            if cluster_edges:
                edge_clusters.append(cluster_edges)

        # Edges not associated with any galaxy pair go into their own single cluster
        # (border-only edges that still need to be placed)
        unassigned = [e for e in missing_edges if e not in edge_to_galaxies]
        if unassigned:
            edge_clusters.append(unassigned)

        return edge_clusters

    # ----------------------------------------------------------
    # CONQUER — recursive greedy solve within one cluster
    # ----------------------------------------------------------

    def _conquer(
        self,
        cluster_edges: List[Tuple[str, int, int]],
        current_edges: Set[Tuple[str, int, int]],
        depth: int
    ) -> Tuple[int, Optional[Tuple[str, int, int]]]:
        """
        Recursive conquer step:
          Base case: 0 or 1 edges in cluster → trivial
          Recursive: score each edge with 3-tier greedy,
                     for top-2 candidates do one level of lookahead
                     (hypothetically place edge, re-divide remaining,
                      and add the best sub-cluster score — limited depth)
        Returns (best_score, best_edge).
        """
        if not cluster_edges:
            return (0, None)
        if len(cluster_edges) == 1:
            return (self._greedy_score(cluster_edges[0], current_edges), cluster_edges[0])

        # Score every edge in this cluster (GREEDY tier scoring)
        scored: List[Tuple[int, Tuple[str, int, int]]] = []
        for e in cluster_edges:
            s = self._greedy_score(e, current_edges)
            scored.append((s, e))

        scored = merge_sort(scored, key=lambda p: p[0], reverse=True)

        # Lookahead: evaluate top candidates one level deeper
        MAX_LOOKAHEAD_DEPTH = 2
        LOOKAHEAD_CANDIDATES = 3  # only recurse on top 3 to keep it fast

        if depth < MAX_LOOKAHEAD_DEPTH:
            refined: List[Tuple[int, Tuple[str, int, int]]] = []
            for i in range(min(LOOKAHEAD_CANDIDATES, len(scored))):
                base_score, candidate = scored[i]
                # Simulate placing this edge
                hypo_edges = current_edges | {candidate}
                remaining  = [e for e in cluster_edges if e != candidate]
                if remaining:
                    # Re-divide remaining after this hypothetical placement
                    sub_clusters = self._divide(remaining, hypo_edges)
                    future_score = 0
                    for sc in sub_clusters:
                        fs, _ = self._conquer(sc, hypo_edges, depth + 1)
                        future_score += fs
                    total = base_score + future_score // 2   # discount future
                else:
                    total = base_score
                refined.append((total, candidate))

            refined = merge_sort(refined, key=lambda p: p[0], reverse=True)
            return refined[0]

        # At max depth: just return best greedy score
        return scored[0]

    # ----------------------------------------------------------
    # GREEDY SCORING — 3-tier system
    # ----------------------------------------------------------

    def _greedy_score(
        self,
        edge: Tuple[str, int, int],
        current_edges: Set[Tuple[str, int, int]]
    ) -> int:
        """
        Tier 1 — COMPLETE (score 1000+):
            Placing this edge finishes a galaxy rectangle right now.
            → massive priority (bonus turn for computer)
            + extra points per cell in that galaxy (bigger = more valuable)

        Tier 2 — SETUP (score 500+):
            After placing this edge, some galaxy will have only 1 missing edge.
            → set up a future completion

        Tier 3 — SEPARATE (BFS score, 1-499):
            BFS impact: does this edge disconnect two adjacent cells?
            → disconnected  : N²+10   (strongly separates two regions)
            → long detour   : detour length  (important structural border)
            → short detour  : small score    (less critical)
        """
        hypo_edges = current_edges | {edge}
        N = self.N

        # ── TIER 1: Does placing this edge complete a galaxy? ──
        gi = self._galaxies_separated_by(edge)
        if gi is not None:
            for g_idx in gi:
                rect = self.puzzle.rects[g_idx]
                if galaxy_is_complete(rect, hypo_edges):
                    size_bonus = rect[2] * rect[3]  # w*h cells
                    return 1000 + size_bonus * 10

        # ── TIER 2: Does this set up a 1-away completion? ──
        if gi is not None:
            for g_idx in gi:
                rect = self.puzzle.rects[g_idx]
                missing_count = self._count_missing_edges_for_galaxy(rect, hypo_edges)
                if missing_count == 1:
                    size_bonus = rect[2] * rect[3]
                    return 500 + size_bonus * 5

        # ── TIER 3: BFS separation score ──
        t, x, y = edge
        if t == 'h':
            if y <= 0 or y >= N:
                return 1
            a_id = (y - 1) * N + x
            b_id =  y      * N + x
        else:
            if x <= 0 or x >= N:
                return 1
            a_id = y * N + (x - 1)
            b_id = y * N +  x

        adj_blocked = self._cell_adj_graph(extra_block=edge, current_edges=current_edges)
        d = bfs_shortest_path(adj_blocked, a_id, b_id)
        if d is None:
            return N * N + 10    # true disconnection — high value
        return max(1, d)         # detour length

    # ----------------------------------------------------------
    # UTILITIES
    # ----------------------------------------------------------

    def _galaxies_separated_by(
        self, edge: Tuple[str, int, int]
    ) -> Optional[Tuple[int, int]]:
        """
        Return the pair (galaxy_index_A, galaxy_index_B) that this edge separates.
        Returns None if edge is on outer border or irrelevant.
        """
        t, x, y = edge
        N = self.N
        if t == 'h':
            if y <= 0 or y >= N:
                return None
            ca = self.puzzle.owner[self.puzzle.cell_id(x, y - 1)]
            cb = self.puzzle.owner[self.puzzle.cell_id(x, y)]
        else:
            if x <= 0 or x >= N:
                return None
            ca = self.puzzle.owner[self.puzzle.cell_id(x - 1, y)]
            cb = self.puzzle.owner[self.puzzle.cell_id(x, y)]

        if ca == cb or ca < 0 or cb < 0:
            return None
        return (ca, cb)

    def _count_missing_edges_for_galaxy(
        self, rect: Rect, current_edges: Set[Tuple[str, int, int]]
    ) -> int:
        """Count how many border edges of this galaxy rectangle are still unplaced."""
        rx, ry, rw, rh = rect
        missing = 0
        for x in range(rx, rx + rw):
            if ('h', x, ry) not in current_edges:
                missing += 1
            if ('h', x, ry + rh) not in current_edges:
                missing += 1
        for y in range(ry, ry + rh):
            if ('v', rx, y) not in current_edges:
                missing += 1
            if ('v', rx + rw, y) not in current_edges:
                missing += 1
        return missing

    def _cell_adj_graph(
        self,
        extra_block: Optional[Tuple[str, int, int]] = None,
        current_edges: Optional[Set[Tuple[str, int, int]]] = None
    ) -> Dict[int, List[int]]:
        """Cell adjacency graph respecting placed + optionally one extra blocked edge."""
        adj: Dict[int, List[int]] = defaultdict(list)
        N = self.N
        blocked = set(current_edges or self.game.edges)
        if extra_block:
            blocked.add(extra_block)

        for y in range(N):
            for x in range(N):
                u = y * N + x
                if x + 1 < N and ('v', x + 1, y) not in blocked:
                    v = y * N + (x + 1)
                    adj[u].append(v); adj[v].append(u)
                if y + 1 < N and ('h', x, y + 1) not in blocked:
                    v = (y + 1) * N + x
                    adj[u].append(v); adj[v].append(u)
        return adj


# ============================================================
# GAME STATE
# ============================================================

@dataclass
class Move:
    edge: Tuple[str, int, int]
    added: bool
    who: str


class GalaxiesGame:
    def __init__(self, n: int = 7, seed: Optional[int] = None):
        self.N   = n
        self.rng = random.Random(seed)
        self.player_score   = 0
        self.computer_score = 0
        self.new_puzzle()

    @staticmethod
    def border_edges(n: int) -> Set[Tuple[str, int, int]]:
        edges: Set[Tuple[str, int, int]] = set()
        for x in range(n):
            edges.add(('h', x, 0)); edges.add(('h', x, n))
        for y in range(n):
            edges.add(('v', 0, y)); edges.add(('v', n, y))
        return edges

    def new_puzzle(self):
        self.puzzle  = GalaxiesPuzzle(n=self.N, rng=self.rng)
        self.puzzle.generate()
        self.fixed   = self.border_edges(self.N)
        self.solution = set(self.puzzle.solution_edges) - self.fixed
        self.player_score   = 0
        self.computer_score = 0
        self._reset_state()

    def reset(self):
        self.player_score   = 0
        self.computer_score = 0
        self._reset_state()

    def _reset_state(self):
        self.edges:      Set[Tuple[str, int, int]]            = set(self.fixed)
        self.edge_owner: Dict[Tuple[str, int, int], str]      = {}
        self.history:    List[Move]                           = []
        self.redo_stack: List[Move]                           = []
        self.turn        = "player"
        self._prev_completed = 0

    def toggle_edge(self, edge: Tuple[str, int, int], who: str) -> bool:
        if edge in self.fixed:
            return False
        if edge in self.edges:
            self.edges.remove(edge)
            self.edge_owner.pop(edge, None)
            self.history.append(Move(edge, added=False, who=who))
        else:
            self.edges.add(edge)
            self.edge_owner[edge] = who
            self.history.append(Move(edge, added=True, who=who))
        self.redo_stack.clear()
        return True

    def check_score_update(self, who: str) -> int:
        """Returns how many NEW galaxies were completed."""
        now = count_completed(self.puzzle, self.edges)
        gained = now - self._prev_completed
        if gained > 0:
            if who == "player":
                self.player_score += gained
            else:
                self.computer_score += gained
            self._prev_completed = now
        return gained

    def undo(self) -> bool:
        if not self.history:
            return False
        mv = self.history.pop()
        if mv.added:
            self.edges.discard(mv.edge)
            self.edge_owner.pop(mv.edge, None)
        else:
            self.edges.add(mv.edge)
            self.edge_owner[mv.edge] = mv.who
        self.redo_stack.append(mv)
        self._prev_completed = count_completed(self.puzzle, self.edges)
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        mv = self.redo_stack.pop()
        if mv.added:
            self.edges.add(mv.edge)
            self.edge_owner[mv.edge] = mv.who
        else:
            self.edges.discard(mv.edge)
            self.edge_owner.pop(mv.edge, None)
        self.history.append(mv)
        self._prev_completed = count_completed(self.puzzle, self.edges)
        return True

    def is_solved(self) -> bool:
        return (self.edges - self.fixed) == self.solution

    def all_edges_placed(self) -> bool:
        return self.solution.issubset(self.edges)

    def cell_adj_graph(self, extra_block: Optional[Tuple[str, int, int]] = None) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = defaultdict(list)
        N = self.N
        blocked = set(self.edges)
        if extra_block:
            blocked.add(extra_block)
        for y in range(N):
            for x in range(N):
                u = y * N + x
                if x + 1 < N and ('v', x + 1, y) not in blocked:
                    v = y * N + (x + 1); adj[u].append(v); adj[v].append(u)
                if y + 1 < N and ('h', x, y + 1) not in blocked:
                    v = (y + 1) * N + x; adj[u].append(v); adj[v].append(u)
        return adj

    def computer_move(self) -> Optional[Tuple[str, int, int]]:
        ai   = DivideConquerAI(self)
        best = ai.best_move()
        if best:
            self.toggle_edge(best, who="computer")
        return best


# ============================================================
# TKINTER UI
# ============================================================

PLAYER_COLOR   = "#1a6fcc"
COMPUTER_COLOR = "#cc2222"
NEUTRAL_COLOR  = "#666666"
FIXED_COLOR    = "#111111"
COMPLETE_FILL  = "#b8f0b8"


class GalaxiesUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Galaxies — D&C + Greedy AI  |  Blue=You  Red=Computer")
        self.configure(bg="#f0f0f0")

        self.game = GalaxiesGame(n=7)

        self.cell     = 60
        self.margin   = 30
        self.wall_w   = 6
        self.dot_r    = 9
        self.snap_tol = 0.18

        N = self.game.N
        cw = self.margin * 2 + self.cell * N

        self.canvas = tk.Canvas(self, width=cw, height=cw, bg="#d8d8d8",
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=7, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        self.status_var = tk.StringVar(value="New puzzle — Your turn (Blue)")
        tk.Label(self, textvariable=self.status_var, anchor="w", bg="#f0f0f0",
                 font=("Helvetica", 11)).grid(
            row=1, column=0, columnspan=7, sticky="we", padx=12)

        self.score_var = tk.StringVar(value="You: 0  |  Computer: 0")
        tk.Label(self, textvariable=self.score_var, anchor="center", bg="#f0f0f0",
                 font=("Helvetica", 14, "bold"), fg="#222").grid(
            row=2, column=0, columnspan=7, sticky="we", padx=12, pady=(0, 4))

        # Legend
        lf = tk.Frame(self, bg="#f0f0f0")
        lf.grid(row=3, column=0, columnspan=7, pady=(0, 4))
        for color, label in [(PLAYER_COLOR, "Your edges"),
                              (COMPUTER_COLOR, "Computer edges (D&C AI)")]:
            tk.Label(lf, text="■", fg=color, bg="#f0f0f0",
                     font=("Helvetica", 16)).pack(side="left", padx=4)
            tk.Label(lf, text=label, bg="#f0f0f0",
                     font=("Helvetica", 11)).pack(side="left", padx=(0, 14))

        # AI explanation label
        ai_text = ("AI: DIVIDE board into independent galaxy clusters  →  "
                   "CONQUER each cluster with 3-tier greedy + lookahead  →  "
                   "COMBINE via Merge Sort")
        tk.Label(self, text=ai_text, anchor="center", bg="#e8e8ff",
                 font=("Helvetica", 9), fg="#444", wraplength=cw).grid(
            row=4, column=0, columnspan=7, sticky="we", padx=12, pady=(0, 6))

        # Buttons
        bf = tk.Frame(self, bg="#f0f0f0")
        bf.grid(row=5, column=0, columnspan=7, pady=6)
        for text, cmd in [("New Game",  self.on_new_game),
                          ("Restart",   self.on_restart),
                          ("Undo",      self.on_undo),
                          ("Redo",      self.on_redo),
                          ("Solve",     self.on_solve),
                          ("Hint",      self.on_hint),
                          ("Quit",      self.destroy)]:
            tk.Button(bf, text=text, command=cmd, width=9,
                      font=("Helvetica", 10)).pack(side="left", padx=4)

        self.redraw()

    # ── coordinate helpers ────────────────────────────────────
    def gx(self, x): return self.margin + x * self.cell
    def gy(self, y): return self.margin + y * self.cell

    # ── drawing ───────────────────────────────────────────────
    def redraw(self):
        self.canvas.delete("all")
        N = self.game.N

        # Grid
        for i in range(N + 1):
            self.canvas.create_line(self.gx(i), self.gy(0),
                                    self.gx(i), self.gy(N), fill="#9a9a9a", width=1)
            self.canvas.create_line(self.gx(0), self.gy(i),
                                    self.gx(N), self.gy(i), fill="#9a9a9a", width=1)

        # Completed galaxy highlights
        for rect in self.game.puzzle.rects:
            if galaxy_is_complete(rect, self.game.edges):
                rx, ry, rw, rh = rect
                self.canvas.create_rectangle(
                    self.gx(rx) + 3, self.gy(ry) + 3,
                    self.gx(rx + rw) - 3, self.gy(ry + rh) - 3,
                    fill=COMPLETE_FILL, outline="")

        # Dots
        for (dx, dy) in self.game.puzzle.dots:
            cx, cy = self.gx(dx), self.gy(dy)
            self.canvas.create_oval(cx - self.dot_r, cy - self.dot_r,
                                    cx + self.dot_r, cy + self.dot_r,
                                    outline="black", width=2, fill="white")

        # Edges with ownership colors
        def ekey(e):
            t, x, y = e
            return (0 if t == 'h' else 1, x, y)

        for e in merge_sort(list(self.game.edges), key=ekey):
            t, x, y = e
            if t == 'h':
                x0, y0, x1, y1 = self.gx(x), self.gy(y), self.gx(x+1), self.gy(y)
            else:
                x0, y0, x1, y1 = self.gx(x), self.gy(y), self.gx(x), self.gy(y+1)
            if e in self.game.fixed:
                color, w = FIXED_COLOR, 8
            else:
                owner = self.game.edge_owner.get(e, "neutral")
                color = (PLAYER_COLOR   if owner == "player"   else
                         COMPUTER_COLOR if owner == "computer" else NEUTRAL_COLOR)
                w = self.wall_w
            self.canvas.create_line(x0, y0, x1, y1, width=w,
                                    fill=color, capstyle=tk.ROUND)

        # Outer border
        self.canvas.create_rectangle(self.gx(0), self.gy(0),
                                     self.gx(N), self.gy(N),
                                     outline=FIXED_COLOR, width=8)

        # Status
        adj   = self.game.cell_adj_graph()
        comps = bfs_components(adj, N * N)
        if self.game.is_solved():
            turn_txt = "Solved! 🎉"
        elif self.game.turn == "player":
            turn_txt = "Your turn (Blue)"
        else:
            turn_txt = "Computer thinking… (Red)"
        self.status_var.set(f"{turn_txt}  |  Open regions: {len(comps)}")
        self.score_var.set(
            f"You: {self.game.player_score}  |  Computer: {self.game.computer_score}")

    # ── click → edge ──────────────────────────────────────────
    def edge_from_click(self, px, py):
        N = self.game.N
        gx = (px - self.margin) / self.cell
        gy = (py - self.margin) / self.cell
        if gx < -0.2 or gy < -0.2 or gx > N + 0.2 or gy > N + 0.2:
            return None
        rx, ry   = round(gx), round(gy)
        dx, dy   = abs(gx - rx), abs(gy - ry)
        if min(dx, dy) > self.snap_tol:
            return None
        if dx < dy:
            x, y = int(rx), int(gy)
            if 0 <= x <= N and 0 <= y < N:
                return ('v', x, y)
        else:
            x, y = int(gx), int(ry)
            if 0 <= x < N and 0 <= y <= N:
                return ('h', x, y)
        return None

    # ── player turn ───────────────────────────────────────────
    def on_click(self, event):
        if self.game.all_edges_placed() or self.game.turn != "player":
            return
        edge = self.edge_from_click(event.x, event.y)
        if edge is None or not self.game.toggle_edge(edge, who="player"):
            return
        gained = self.game.check_score_update("player")
        self.redraw()
        if self.game.all_edges_placed():
            self.show_result(); return
        if gained > 0:
            self.status_var.set(
                f"You completed {gained} galaxy! Bonus turn! (Blue)")
            self.redraw()
            # bonus turn stays with player — do nothing, wait for next click
        else:
            self.game.turn = "computer"
            self.redraw()
            self.after(250, self.do_computer_turn)

    # ── computer turn ─────────────────────────────────────────
    def do_computer_turn(self):
        if self.game.all_edges_placed():
            self.show_result(); return
        best = self.game.computer_move()
        if best is None:
            self.game.turn = "player"; self.redraw(); return
        gained = self.game.check_score_update("computer")
        self.redraw()
        if self.game.all_edges_placed():
            self.show_result(); return
        if gained > 0:
            self.status_var.set(
                f"Computer completed {gained} galaxy! Bonus turn… (Red)")
            self.redraw()
            self.after(500, self.do_computer_turn)   # computer bonus turn
        else:
            self.game.turn = "player"
            self.redraw()

    def show_result(self):
        p, c = self.game.player_score, self.game.computer_score
        result = ("🎉 You win!" if p > c else
                  "Computer wins!" if c > p else "It's a tie!")
        self.status_var.set(f"Game Over — {result}  ({p} vs {c})")
        messagebox.showinfo("Game Over",
            f"{result}\n\nYou: {p} galaxies\nComputer: {c} galaxies")

    # ── buttons ───────────────────────────────────────────────
    def on_new_game(self):
        self.game.new_puzzle(); self.redraw()

    def on_restart(self):
        self.game.reset(); self.redraw()

    def on_undo(self):
        if self.game.undo():
            self.game.turn = "player"; self.redraw()

    def on_redo(self):
        if self.game.redo():
            self.game.turn = "player"; self.redraw()

    def on_solve(self):
        for e in self.game.solution:
            if e not in self.game.edges:
                self.game.edges.add(e)
                self.game.edge_owner[e] = "neutral"
        self.game.turn = "player"
        self.game._prev_completed = count_completed(self.game.puzzle, self.game.edges)
        self.redraw()
        messagebox.showinfo("Galaxies", "Solution drawn.")

    def on_hint(self):
        if self.game.all_edges_placed() or self.game.turn != "player":
            return
        self.game.turn = "computer"
        self.redraw()
        self.after(120, self.do_computer_turn)


# ============================================================
if __name__ == "__main__":
    GalaxiesUI().mainloop()
