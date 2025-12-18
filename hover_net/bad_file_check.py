import os, cv2, glob

inp = "/project/KidneyHE/data_lung/P15_LUAD/histocell/tiles/"
bad = []
for p in sorted(glob.glob(os.path.join(inp, "*"))):
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        bad.append((p, os.path.exists(p), os.path.getsize(p) if os.path.exists(p) else -1))
print("Bad count:", len(bad))
for p, exists, size in bad[:50]:
    print(p, "exists=", exists, "size=", size)
