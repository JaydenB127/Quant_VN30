# -*- coding: utf-8 -*-
with open("ets/api/routes/views.py", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if "lucide.createIcons" in line:
            clean_line = line.strip().encode("ascii", "ignore").decode("ascii")
            print(f"{i+1}: {clean_line}")
