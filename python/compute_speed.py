import os
import sys
import os
import re
from collections import defaultdict

buffer_check = re.compile(r"train:\s*([\d\.]+)(.*?)buffer_add:\s*([\d\.]+)")
search_check = re.compile(r"search_ratio:\s*([\d\.]+)")

result = defaultdict(lambda: list())

root = sys.argv[1]
_, subfolders, files = next(os.walk(root))

for subfolder in subfolders:
    if subfolder.startswith("."):
        continue

    print(f"Dealing with {subfolder}")
    avg_train_speed = 0
    avg_buffer_add_speed = 0
    n = 0
    search_ratio = -1.0

    for line in open(os.path.join(root, subfolder, "main2.log")):
        m = buffer_check.search(line) 
        if m:
            avg_train_speed += float(m.group(1))
            avg_buffer_add_speed += float(m.group(3))
            n += 1
            continue

        m = search_check.search(line) 
        if m:
            search_ratio = float(m.group(1))

    if search_ratio < 0 or n == 0:
        print(f"Invalid file, skipping.. search_ratio = {search_ratio}, n = {n}")
    else:
        result[search_ratio].append((avg_train_speed/n, avg_buffer_add_speed/n))


for k, v in result.items():
    train_speed = [ a for a, b in v ]
    buffer_add_speed = [ b for a, b in v ]

    print(f"{k}: {sum(train_speed) / len(v)}, {sum(buffer_add_speed) / len(v)}")

