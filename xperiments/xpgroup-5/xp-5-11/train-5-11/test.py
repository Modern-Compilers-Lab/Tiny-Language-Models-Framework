import time
for i in range(1000):
	t0 = time.time()
	print("Hello")
	t1 = time.time()
	print(f"RATE: {1/(t1-t0)} | SPEED: {t1-t0} sec/it.")