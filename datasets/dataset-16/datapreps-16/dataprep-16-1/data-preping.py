## Data preping (on Kindi)

## Imports
import pickle
import numpy as np
import gc
from tinypy_tokenizer import TinypyTokenizer

## Logging boilerplate
log_file = open("data-preping.log", "w")
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


## Saving the numpy random state
log("Saving the data-preping-numpy-random state")
log("saving it")
np_random_state = np.random.get_state()
with open("data-preping-np-random-state.bin", "wb") as f:
	pickle.dump(np_random_state, f)
log("freeing its memory")
del np_random_state
gc.collect()

# We create the TinypyTokenizer instance
tpt = TinypyTokenizer()
# We generate the tokenized file of data.txt
log("We generate the tokenized file of data.txt directly to train.bin")
print(tpt.encode_to_file("../../data-ds-16/train.txt", "data-dp-16-1/train.bin"))
print(tpt.encode_to_file("../../data-ds-16/val.txt", "data-dp-16-1/val.bin"))

log_file.close()