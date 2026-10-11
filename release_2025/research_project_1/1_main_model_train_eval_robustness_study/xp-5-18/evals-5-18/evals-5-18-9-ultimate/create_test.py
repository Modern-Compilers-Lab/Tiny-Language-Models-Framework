import pandas as pd
import re
import random
import math
from tqdm import tqdm


df = pd.read_csv("/scratch/yb2618/Tiny-Language-Models-Framework/xperiments/xpgroup-5/xp-5-18/evals-5-18/evals-5-18-10/infers/hard-match-successes.csv")



inputs = [i for i in range(len(df["example_input"]))]
random.shuffle(inputs)

r_assign = re.compile("(\t*)([a-z])( = )([a-z]|(?:-*\d*))")
r_arith = re.compile("(\t*)([a-z])( = )([a-z]|(?:-*\d*))( .{,2} )([a-z]|(?:-*\d*))")
r_while = re.compile("(\t*while )([a-z]|(?:-*\d*))( .{,2} )([a-z]|(?:-*\d*))(:)")
r_if = re.compile("(\t*(?:elif|if) )([a-z]|(?:-*\d*))( .{,2} )([a-z]|(?:-*\d*))(:)")
r_print = re.compile("((?:\t*)print\()([a-z])(\))")
r_print_l0 = re.compile("^print\(([a-z]|(\d)*)\)$")
r = re.compile("\t*([a-z]) = ([a-z]|(?:-*\d*)).*")

f = open("./test.txt", "w")
all_vars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
new_snippets = list()

for i in tqdm(inputs):
	snippet = df["example_input"][i].split("\n")[0:-2]
	## snippet vars
	snippet_vars = set()
	for line in snippet:
		m = r.match(line)
		if m:
			snippet_vars.add(m.group(1))
	snippet_vars = list(snippet_vars)
	
	combs = list()
	combs_set = set()

	# We only keep 10 of them
	while len(combs) < 128 :
		new_comb = random.sample(all_vars, len(snippet_vars))
		if s := str(new_comb) in combs_set:
			continue
		else:
			combs_set.add(s)
			combs.append(new_comb)
	
	for comb in combs:
		new_snippet = list(snippet)
		permutations = {k: v for k, v in zip(snippet_vars, comb)}
		for i, line in enumerate(new_snippet):
			if (m := r_arith.match(line)) or (m := r_assign.match(line)) or (m := r_while.match(line)) or (m := r_if.match(line)) or (m := r_print.match(line)):
				groups = list(m.groups())
				for j, _ in enumerate(groups):
					if _ in permutations:
						groups[j] = permutations[_]
				new_snippet[i] = ''.join(groups)
		new_snippets.append('\n'.join(new_snippet) + "\n# output\n# 1\n\n")
	
f.write(''.join(list(new_snippets)))
print(len(new_snippets))