#!/usr/bin/env python3

#!/usr/bin/env python3

import re
import sys

filename = sys.argv[1] if len(sys.argv) > 1 else "time_log"

total_minutes = 0

with open(filename, "r") as f:
    for line in f:
        m = re.search(r'(\d+)h(\d{2})', line)
        if m:
            hours = int(m.group(1))
            minutes = int(m.group(2))
            total_minutes += hours * 60 + minutes

hours = total_minutes // 60
minutes = total_minutes % 60

print(f"Total: {hours}h{minutes:02d} ({total_minutes} minutes)")


if __name__ == "__main__":
    ...
