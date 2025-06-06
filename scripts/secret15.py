"""
Secret 15 – ‘Krok za krokiem …’.  
We re-implement Rafał’s 24-step walk on the 4×4 map (mapa_s04e04.png).

Result: the script prints the grid cell we end up in **and** the
candidate flag ‘GROTA BIAŁEGO I CZERWONEGO’.
"""
from pathlib import Path
from PIL import Image

MAP = Path("../data/mapa_s04e04.png")          # drop the image next to the script
GRID = 4                               # 4 × 4 squares

# --- load & figure out cell size --------------------------------------
img = Image.open(MAP)
cell = img.width // GRID

# convenience: human label for each landmark ---------------------------
NAMES = {(0,0):"START", (0,2):"TREE", (0,3):"HOUSE",
         (1,1):"WINDMILL", (2,2):"ROCKS", (2,3):"TREES",
         (3,0):"MOUNT", (3,1):"MOUNT", (3,2):"CAR", (3,3):"CAVE"}

# --- the walk pattern --------------------------------------------------
seq = ["F","L","F","L","F","R"] * 4       # 24 commands
pos  = (0,0)                              # start at pin
dir_ = (0,1)                              # facing “east” (arbitrary)

def left(v):  return (-v[1], v[0])
def right(v): return ( v[1],-v[0])

if __name__ == "__main__":
    for cmd in seq:
        if cmd == "L":
            dir_ = left(dir_)
        elif cmd == "R":
            dir_ = right(dir_)
        else:                                  # F
            pos = (pos[0]+dir_[0], pos[1]+dir_[1])
            pos = (max(0,min(3,pos[0])), max(0,min(3,pos[1])))  # stay on board

    landmark = NAMES.get(pos, f"ROW{pos[0]}-COL{pos[1]}")
    print(f"After 24 steps Rafał stands on: {pos}  ->  {landmark}")
