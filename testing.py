import subprocess
import time
import threading
import signal

num = 128
while (num <= 4194304):
    timings = []
    print(f"Running simulation with {num} boids...")
    try:
        result = subprocess.run(['build/bin/release/cis5650_boids.exe', f'{num}'], capture_output=True, text=True)
        stdout = result.stdout
    except subprocess.TimeoutExpisred as e:
        stdout = e.stdout 
    for line in stdout.strip().splitlines():
        try:
            timings.append(float(line))
        except ValueError:
            continue
    if (len(timings) == 0):
        print(f"No valid timing data found for {num} boids.")
    else:
        print(f"Timings for {num} boids: {sum(timings) / len(timings)} ms")
    num *= 2
