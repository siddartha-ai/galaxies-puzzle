
import tkinter as tk
from tkinter import messagebox
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import random

def merge_sort(arr: List, key=lambda x: x, reverse: bool = False) -> List:
    """Return a new list sorted by key using Merge Sort (no built-in sorting)."""
    n = len(arr)
    if n <= 1:
        return arr[:]

    mid = n // 2
    left = merge_sort(arr[:mid], key=key, reverse=reverse)
    right = merge_sort(arr[mid:], key=key, reverse=reverse)

    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        a = key(left[i])
        b = key(right[j])
        if reverse:
            take_left = a >= b
        else:
            take_left = a <= b
        if take_left:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    while i < len(left):
        merged.append(left[i])
        i += 1
    while j < len(right):
        merged.append(right[j])
        j += 1

    return merged


def bfs_shortest_path(adj: Dict[int, List[int]], start: int, goal: int) -> Optional[int]:
 
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

    seen = set()
    comps = []
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


Rect = Tuple[int, int, int, int]


class GalaxiesPuzzle:

    def __init__(self, n: int = 7, rng: Optional[random.Random] = None):
        self.N = n
        self.rng = rng or random.Random()
        self.rects: List[Rect] = []
        self.owner: List[int] = [-1] * (n * n)
        self.dots: List[Tuple[float, float]] = []
        self.solution_edges: Set[Tuple[str, int, int]] = set()

    def cell_id(self, x: int, y: int) -> int:
        return y * self.N + x

    def generate(self, target_rects: Optional[int] = None) -> None:

        n = self.N
        if target_rects is None:
            target_rects = self.rng.randint(9, 14)

        rects: List[Rect] = [(0, 0, n, n)]

        def can_split(r: Rect) -> bool:
            _, _, w, h = r
            return (w >= 2) or (h >= 2)

        tries = 0
        while len(rects) < target_rects and tries < 5000:
            tries += 1
            candidates = [r for r in rects if can_split(r)]
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
                r1 = (x, y, k, h)
                r2 = (x + k, y, w - k, h)
            else:
                k = self.rng.randint(1, h - 1)
                r1 = (x, y, w, k)
                r2 = (x, y + k, w, h - k)

            rects.append(r1)
            rects.append(r2)

        self.rects = rects

        self.owner = [-1] * (n * n)
        for idx, (x, y, w, h) in enumerate(self.rects):
            for yy in range(y, y + h):
                for xx in range(x, x + w):
                    self.owner[self.cell_id(xx, yy)] = idx

        self.dots = []
        for (x, y, w, h) in self.rects:
            self.dots.append((x + w / 2.0, y + h / 2.0))

        self.solution_edges = self.compute_solution_edges()

    def compute_solution_edges(self) -> Set[Tuple[str, int, int]]:

        n = self.N
        edges: Set[Tuple[str, int, int]] = set()


        for x in range(n):
            edges.add(('h', x, 0))
            edges.add(('h', x, n))
        for y in range(n):
            edges.add(('v', 0, y))
            edges.add(('v', n, y))

        for y in range(n):
            for x in range(n):
                o = self.owner[self.cell_id(x, y)]
                if x + 1 < n:
                    o2 = self.owner[self.cell_id(x + 1, y)]
                    if o2 != o:
                        edges.add(('v', x + 1, y))
                if y + 1 < n:
                    o2 = self.owner[self.cell_id(x, y + 1)]
                    if o2 != o:
                        edges.add(('h', x, y + 1))
        return edges


@dataclass
class Move:
    edge: Tuple[str, int, int]
    added: bool
    who: str


class GalaxiesGame:
    def __init__(self, n: int = 7, seed: Optional[int] = None):
        self.N = n
        self.rng = random.Random(seed)
        self.new_puzzle()

    @staticmethod
    def border_edges(n: int) -> Set[Tuple[str, int, int]]:
        edges = set()
        for x in range(n):
            edges.add(('h', x, 0))
            edges.add(('h', x, n))
        for y in range(n):
            edges.add(('v', 0, y))
            edges.add(('v', n, y))
        return edges

    def new_puzzle(self):
        self.puzzle = GalaxiesPuzzle(n=self.N, rng=self.rng)
        self.puzzle.generate()
        self.fixed = self.border_edges(self.N)
        self.solution = set(self.puzzle.solution_edges) - set(self.fixed)
        # Precompute edges per galaxy (Divide & Conquer idea)
        from collections import defaultdict
        self.galaxy_edges = defaultdict(set)

        for e in self.solution:
            t, x, y = e
            cells = (
                [(x, y - 1), (x, y)] if t == 'h'
                else [(x - 1, y), (x, y)]
            )

            for cx, cy in cells:
                if 0 <= cx < self.N and 0 <= cy < self.N:
                    gid = self.puzzle.owner[cy * self.N + cx]
                    self.galaxy_edges[gid].add(e)

        self.reset()

    def reset(self):
        self.edges: Set[Tuple[str, int, int]] = file_edges_copy(self.fixed)
        self.history: List[Move] = []
        self.redo_stack: List[Move] = []
        self.turn = "player"
        self.anchor_stack = []
        self.cached_components = None



    def edge_hits_dot(self, edge: Tuple[str, int, int]) -> bool:

        t, x, y = edge
        eps = 1e-9

        for dx, dy in self.puzzle.dots:
            if t == 'h':
              
                if abs(dy - y) < eps and (x - eps) <= dx <= (x + 1 + eps):
                    return True
            else:

                if abs(dx - x) < eps and (y - eps) <= dy <= (y + 1 + eps):
                    return True

        return False
    def toggle_edge(self, edge: Tuple[str, int, int], who: str) -> bool:
        if edge in self.fixed:
            return False

        if self.edge_hits_dot(edge):
            return False

        added = False

        if edge in self.edges:
            self.edges.remove(edge)
            self.history.append(Move(edge=edge, added=False, who=who))
        else:
            self.edges.add(edge)
            self.history.append(Move(edge=edge, added=True, who=who))
            added = True

        if added and who in ("player", "computer"):
            self.anchor_stack.append(edge)

        self.redo_stack.clear()

        self.cached_components = None

        return True





    def undo(self) -> bool:
        if not self.history:
            return False
        mv = self.history.pop()
        if mv.added:
            self.edges.discard(mv.edge)
        else:
            self.edges.add(mv.edge)
        self.redo_stack.append(mv)
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        mv = self.redo_stack.pop()
        if mv.added:
            self.edges.add(mv.edge)
        else:
            self.edges.discard(mv.edge)
        self.history.append(mv)
        return True

    def is_solved(self) -> bool:
        return (self.edges - self.fixed) == self.solution

    def cell_adj_graph(self, extra_block: Optional[Tuple[str, int, int]] = None) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = defaultdict(list)
        n = self.N
        blocked = set(self.edges)
        if extra_block is not None:
            blocked.add(extra_block)

        def cid(x: int, y: int) -> int:
            return y * n + x

        for y in range(n):
            for x in range(n):
                u = cid(x, y)
                if x + 1 < n:
                    w = ('v', x + 1, y)
                    if w not in blocked:
                        v = cid(x + 1, y)
                        adj[u].append(v)
                        adj[v].append(u)
                if y + 1 < n:
                    w = ('h', x, y + 1)
                    if w not in blocked:
                        v = cid(x, y + 1)
                        adj[u].append(v)
                        adj[v].append(u)
        return adj
    def galaxy_from_edge(self, edge):
        t, x, y = edge
        if t == 'h':
            cx, cy = x, y - 1
        else:
            cx, cy = x - 1, y
        if 0 <= cx < self.N and 0 <= cy < self.N:
            return self.puzzle.owner[cy * self.N + cx]
        return None
    def galaxy_completed(self, galaxy_id):
        return self.galaxy_edges[galaxy_id] <= self.edges

    def nearest_unfinished_galaxy(self, start_cell):
        adj = self.cell_adj_graph()
        q = deque([start_cell])
        seen = {start_cell}

        while q:
            u = q.popleft()
            gid = self.puzzle.owner[u]

            if not self.galaxy_completed(gid):
                return gid

            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return None
    def edges_touch(self, e1, e2):
        t1, x1, y1 = e1
        t2, x2, y2 = e2

        p1 = {(x1, y1), (x1 + (t1 == 'h'), y1 + (t1 == 'v'))}
        p2 = {(x2, y2), (x2 + (t2 == 'h'), y2 + (t2 == 'v'))}

        return bool(p1 & p2)

    def best_wall_in_galaxy(self, galaxy_id, anchor=None, require_touch=True):

        remaining = self.solution - self.edges
        candidates = []

        for e in remaining:
            t, x, y = e
            cells = (
                [(x, y - 1), (x, y)] if t == 'h'
                else [(x - 1, y), (x, y)]
            )

            for cx, cy in cells:
                if 0 <= cx < self.N and 0 <= cy < self.N:
                    if self.puzzle.owner[cy * self.N + cx] == galaxy_id:
                        candidates.append(e)
                        break

        if not candidates:
            return None

        def edge_key(e):
            t, x, y = e
            return (0 if t == 'h' else 1, x, y)

        candidates = merge_sort(candidates, key=edge_key)

        if require_touch and anchor is not None:
            for e in candidates:
                if self.edges_touch(anchor, e):
                    return e
            return None  
        return candidates[0]



    def any_unfinished_galaxy(self):
        for gid in range(len(self.puzzle.dots)):
            if not self.galaxy_completed(gid):
                return gid
        return None
    def greedy_best_wall(self, galaxy_id):
        remaining = self.galaxy_edges[galaxy_id] - self.edges

        best_edge = None
        best_score = -1

        for e in remaining:
            # Temporarily add
            self.edges.add(e)

            score = 0

            # Big reward if completes galaxy
            if self.galaxy_completed(galaxy_id):
                score += 100

            # Small reward for touching existing walls
            for existing in self.edges:
                if existing != e and self.edges_touch(existing, e):
                    score += 1

            # Remove temporary
            self.edges.remove(e)

            if score > best_score:
                best_score = score
                best_edge = e

        return best_edge

    def computer_move(self):

        # Create list of unfinished galaxies
        unfinished = []

        for gid in range(len(self.puzzle.dots)):
            if not self.galaxy_completed(gid):
                remaining = len(self.galaxy_edges[gid] - self.edges)
                unfinished.append((remaining, gid))

        if not unfinished:
            return None

        unfinished = merge_sort(unfinished, key=lambda x: x[0])
        _, best_gid = unfinished[0]

        move = self.greedy_best_wall(best_gid)
        if move:
            self.toggle_edge(move, "computer")
            return move

        return None






def file_edges_copy(s: Set[Tuple[str, int, int]]) -> Set[Tuple[str, int, int]]:

    return set(s)


class GalaxiesUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Random Galaxies - Player vs Computer (No built-in sorting)")

        self.game = GalaxiesGame(n=7)

        self.cell = 60
        self.margin = 30
        self.wall_w = 6
        self.grid_w = 1
        self.dot_r = 9
        self.snap_tol = 0.18

        n = self.game.N
        w = self.margin * 2 + self.cell * n
        h = self.margin * 2 + self.cell * n

        self.canvas = tk.Canvas(self, width=w, height=h, bg="#d8d8d8", highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=7, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)

        self.status = tk.StringVar(value="New random puzzle generated. Your turn.")
        tk.Label(self, textvariable=self.status, anchor="w").grid(row=1, column=0, columnspan=7, sticky="we", padx=10)

        tk.Button(self, text="New Game", command=self.on_new_game).grid(row=2, column=0, padx=5, pady=8, sticky="we")
        tk.Button(self, text="Restart", command=self.on_restart).grid(row=2, column=1, padx=5, pady=8, sticky="we")
        tk.Button(self, text="Undo", command=self.on_undo).grid(row=2, column=2, padx=5, pady=8, sticky="we")
        tk.Button(self, text="Redo", command=self.on_redo).grid(row=2, column=3, padx=5, pady=8, sticky="we")
        tk.Button(self, text="Solve", command=self.on_solve).grid(row=2, column=4, padx=5, pady=8, sticky="we")
        tk.Button(self, text="Hint (Computer Move)", command=self.on_hint).grid(row=2, column=5, padx=5, pady=8, sticky="we")
        tk.Button(self, text="Quit", command=self.destroy).grid(row=2, column=6, padx=5, pady=8, sticky="we")

        self.redraw()

    def gx(self, x: float) -> float:
        return self.margin + x * self.cell

    def gy(self, y: float) -> float:
        return self.margin + y * self.cell

    def redraw(self):
        self.canvas.delete("all")
        n = self.game.N

        for i in range(n + 1):
            x = self.gx(i)
            self.canvas.create_line(x, self.gy(0), x, self.gy(n), width=self.grid_w, fill="#9a9a9a")
            y = self.gy(i)
            self.canvas.create_line(self.gx(0), y, self.gx(n), y, width=self.grid_w, fill="#9a9a9a")


        edges_list = list(self.game.edges)

        def edge_key(e: Tuple[str, int, int]) -> Tuple[int, int, int]:
            t, x, y = e
            tval = 0 if t == 'h' else 1
            return (tval, x, y)

        edges_sorted = merge_sort(edges_list, key=edge_key, reverse=False)

        for (t, x, y) in edges_sorted:
            if t == 'h':
                x0, y0 = self.gx(x), self.gy(y)
                x1, y1 = self.gx(x + 1), self.gy(y)
            else:
                x0, y0 = self.gx(x), self.gy(y)
                x1, y1 = self.gx(x), self.gy(y + 1)
            self.canvas.create_line(x0, y0, x1, y1, width=self.wall_w, fill="black", capstyle=tk.ROUND)

        for (dx, dy) in self.game.puzzle.dots:
            cx = self.gx(dx)
            cy = self.gy(dy)
            self.canvas.create_oval(cx - self.dot_r, cy - self.dot_r, cx + self.dot_r, cy + self.dot_r,
                                    outline="black", width=2, fill="white")

        self.canvas.create_rectangle(self.gx(0), self.gy(0), self.gx(n), self.gy(n),
                                     outline="black", width=8)

        if self.game.cached_components is None:
            adj = self.game.cell_adj_graph()
            self.game.cached_components = bfs_components(adj, n * n)

        comps = self.game.cached_components

        user_edges = len(self.game.edges - self.game.fixed)

        msg = ("Your turn" if self.game.turn == "player" else "Computer's turn")
        msg += f" | Lines placed: {user_edges} | Open regions: {len(comps)}"
        if self.game.is_solved():
            msg = "Solved! 🎉 (Lines match the generated solution.)"
        self.status.set(msg)

    def edge_from_click(self, px: int, py: int) -> Optional[Tuple[str, int, int]]:
        n = self.game.N
        gx = (px - self.margin) / self.cell
        gy = (py - self.margin) / self.cell
        if gx < -0.2 or gy < -0.2 or gx > n + 0.2 or gy > n + 0.2:
            return None

        rx, ry = round(gx), round(gy)
        dx, dy = abs(gx - rx), abs(gy - ry)
        if min(dx, dy) > self.snap_tol:
            return None

        if dx < dy:
            x = int(rx)
            y = int(gy)
            if 0 <= x <= n and 0 <= y < n:
                return ('v', x, y)
        else:
            x = int(gx)
            y = int(ry)
            if 0 <= x < n and 0 <= y <= n:
                return ('h', x, y)
        return None

    def on_click(self, event):
        if self.game.is_solved() or self.game.turn != "player":
            return

        edge = self.edge_from_click(event.x, event.y)
        if edge is None:
            return

        if not self.game.toggle_edge(edge, who="player"):
            return

        self.game.turn = "computer"
        self.redraw()
        self.after(120, self.do_computer_turn)

    def do_computer_turn(self):
        if not self.game.is_solved():
            self.game.computer_move()
        self.game.turn = "player"
        self.redraw()
        if self.game.is_solved():
            messagebox.showinfo("Galaxies", "Solved! 🎉")

    def on_new_game(self):
        self.game.new_puzzle()
        self.redraw()

    def on_restart(self):
        self.game.reset()
        self.redraw()

    def on_solve(self):
        self.game.edges = set(self.game.fixed) | set(self.game.solution)
        self.game.turn = "player"
        self.redraw()
        messagebox.showinfo("Galaxies", "Random puzzle generated and solution drawn.")

    def on_hint(self):
        if self.game.is_solved() or self.game.turn != "player":
            return
        self.game.turn = "computer"
        self.redraw()
        self.after(120, self.do_computer_turn)

    def on_undo(self):
        if self.game.undo():
            self.game.turn = "player"
            self.redraw()

    def on_redo(self):
        if self.game.redo():
            self.game.turn = "player"
            self.redraw()


if __name__ == "__main__":
    GalaxiesUI().mainloop()

