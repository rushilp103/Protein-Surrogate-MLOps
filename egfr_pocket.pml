load data/mutants/T790M.pdb, kinase
load data/docking/T790M_docked.pdbqt, ligand
hide all
show cartoon, kinase
show surface, kinase
set transparency, 0.55, kinase
color gray80, kinase
select anchors, resi 790+745+855
show sticks, anchors
color cyan, anchors
show sticks, ligand
util.cbay ligand
zoom anchors, 8
bg_color white
ray 1600, 1000
png outputs/figures/egfr_gefitinib_pocket.png, dpi=200