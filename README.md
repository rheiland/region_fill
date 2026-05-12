# region_fill

```
$ python polygon_boundaries3.py concave.png
Loaded 'concave.png'  shape=(480, 1414, 3)  dtype=uint8
Orange segmentation → 2 region(s)

Region        Verts  Area(px²)     Reg centroid (x,y)   In?         Pole (x,y)   In?   MaxDist
───────────────────────────────────────────────────────────────────────────────────────────────
------>  poly1.csv
Region 1         65      80888   (1059.9, 223.4)         yes   (1094.0, 230.0)   yes      78.4px
------>  poly2.csv
Region 2         39      76962   ( 327.8, 173.6)         yes   ( 328.0, 252.0)   yes     101.8px
Saved rwh_boundaries.png
Saved rwh_centroids.png

── Point-in-polygon test (neighbourhood around pole) ────────────────

  Region 1 — pole=(230,1094), sampled 49 points, 49/49 agree
    geom inside: 49  |  mask inside: 49  |  mismatches: 0

  Region 2 — pole=(252,328), sampled 49 points, 49/49 agree
    geom inside: 49  |  mask inside: 49  |  mismatches: 0

Saved rwh_pip_test.png
```
