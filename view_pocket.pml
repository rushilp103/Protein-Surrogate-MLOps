# 1. Load your minimized mutant structure
load .\data\mutants\T790M.pdb, kinase

# 2. Make the protein a transparent surface so we can see inside the pocket
hide all
show cartoon, kinase
show surface, kinase
set transparency, 0.6, kinase
color gray80, kinase

# 3. Highlight the 3 anchor residues used to triangulate the void
# T790 (Gatekeeper), K745 (Catalytic Lysine), D855 (DFG Motif)
select anchors, resi 790+745+855
show sticks, anchors
color cyan, anchors

# 4. Drop a RED sphere at the BAD center (T790 C-alpha atom)
pseudoatom bad_center, pos=[-11.55, 28.00, 11.18]
show spheres, bad_center
color red, bad_center
set sphere_scale, 1.5, bad_center

# 5. Drop a GREEN sphere at the GOOD center (Calculated true void)
pseudoatom good_center, pos=[-7.21, 30.66, 11.75]
show spheres, good_center
color green, good_center
set sphere_scale, 1.5, good_center

# 6. Center the camera perfectly on the pocket
zoom anchors, 10