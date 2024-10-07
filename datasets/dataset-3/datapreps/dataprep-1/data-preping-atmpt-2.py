## Data preping (on Kindi)
DIR = "/data/yb2618/Tiny-Language-Models-Framework/datasets/dataset-3/datapreps/dataprep-1/"


## Imports
import pickle
import struct
import time
import os

## Logging boilerplate
log_file = open(DIR+"data-preping-atmpt-2.log", "w")
pbar_recept_string = " " * 200 + "\n"
log_file.write(pbar_recept_string)
log_file.write(pbar_recept_string)
log_file.flush()
def log(s:str, p_level=None):
	if p_level == 1:
		log_file.seek(0,0)
		log_file.write(pbar_recept_string)
		log_file.seek(0,0)
		log_file.write(s)
		log_file.seek(0,2)
	elif p_level == 2:
		log_file.seek(len(pbar_recept_string), 0)
		log_file.write(pbar_recept_string)
		log_file.seek(len(pbar_recept_string), 0)
		log_file.write(s)
		log_file.seek(0,2)
	else:
		if s[0].upper() == s[0]:
			start = "\n"
			end = ":"
		else:
			start = "	--> "
			end = ""
		log_file.write(start + s + end + "\n")
	log_file.flush()


## Convert seconds to days, hours, minutes, seconds
def convert_seconds(seconds:float):
	# ignoring the sub seconds 
	seconds = int(seconds)
	days, seconds = divmod(seconds, 86400)
	hours, seconds = divmod(seconds, 3600)
	minutes, seconds = divmod(seconds, 60)
	return (days, hours, minutes, seconds)


## We read the stoi from the meta.pkl
with open(DIR+"data/meta.pkl", "rb") as f:
	meta = pickle.load(f)
stoi = meta["stoi"]
del meta


## Getting back the number of tokens in the .txt files
log("Getting back the number of tokens in the .txt files")
train_tokens = os.path.getsize(DIR+"data/train.txt")
log(f"train_tokens {train_tokens}")
val_tokens = os.path.getsize(DIR+"data/val.txt")
log(f"val_tokens {val_tokens}")

## We define the encoding function
log("We define the encoding function")
def encode_generator(s:str):
	for c in s:
		yield stoi[c]

log("Reading and encoding train.txt directly to binary")
with open(DIR+"data/train.txt", "r") as f, open(DIR+"data/train.bin", "wb") as bin_file:
	chunk_size = 1024 * 1024 * 1000  # 1 GB
	max_iters = int(train_tokens/chunk_size)
	i = 0
	while True:
		past = time.time()
		chunk = f.read(chunk_size)
		if not chunk:
			break
		for token in encode_generator(chunk):
			# past2 = time.time()
			bin_file.write(struct.pack('H', token))  # 'H' stands for unsigned short (2 bytes)
			# present2 = time.time()
			# log(f"|ITERS: {j+1} / {chunk_size:,} | COMP: {(j+1)/chunk_size * 100:.2f}% | RATE: {1/(present2-past2):.2f} it./s | SPD: {present2 - past2 :.4f} s/it.| ERT: {convert_seconds((chunk_size-j-1) * (present2-past2))}", p_level = 2)

		i = i+1
		present = time.time()
		#>>>ERROR HAPPENED HERE: forgot to replace iter with i
		log(f"|ITERS: {i+1} / {max_iters} | COMP: {(i+1)/max_iters * 100:.2f}% | RATE: {1/(present-past):.2f} it./s | SPD: {present - past :.4f} s/it.| ERT: {convert_seconds((max_iters-i-1) * (present-past))}", p_level = 1)

log("Reading and encoding val.txt directly to binary")
with open(DIR+"data/val.txt", "r") as f, open(DIR+"data/val.bin", "wb") as bin_file:
	chunk_size = 1024 * 1024 * 1000  # 1 GB
	max_iters = int(val_tokens/chunk_size)
	i = 0
	while True:
		past = time.time()
		chunk = f.read(chunk_size)
		if not chunk:
			break
		for token in encode_generator(chunk):
			bin_file.write(struct.pack('H', token))  # 'H' stands for unsigned short (2 bytes)
		i = i+1
		present = time.time()
		##>> MISTAKE: here I should just use i not i+1
		log(f"|ITERS: {i+1} / {max_iters} | COMP: {(i+1)/max_iters * 100:.2f}% | RATE: {1/(present-past):.2f} it./s | SPD: {present - past :.4f} s/it.| ERT: {convert_seconds((max_iters-i-1) * (present-past))}", p_level = 2)

log_file.close()