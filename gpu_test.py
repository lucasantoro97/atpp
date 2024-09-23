import cupy as cp

# Get free and total memory
free_mem, total_mem = cp.cuda.runtime.memGetInfo()
used_mem = total_mem - free_mem

print(f"Total GPU Memory: {total_mem / (1024**3):.2f} GB")
print(f"Used GPU Memory: {used_mem / (1024**3):.2f} GB")
print(f"Free GPU Memory: {free_mem / (1024**3):.2f} GB")
