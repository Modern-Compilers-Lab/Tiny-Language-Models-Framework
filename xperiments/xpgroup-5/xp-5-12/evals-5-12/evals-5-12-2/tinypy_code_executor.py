from tqdm import tqdm
from io import StringIO
from contextlib import redirect_stdout


with open('30PS_chatGPT_norm_wo_desc.txt', 'r') as f:
	data = f.read()

examples = data.split('\n\n')[:-1]

f = open('./doutput_30PS_chatGPT_normalized.txt', 'w')

for code in tqdm(examples):
	# i removed the overflow control conditional in this boilerplate ...
	exec_env_boilerplate = f"""
from sys import settrace

def line_tracer(frame, event, arg):
	if event == "exception" :
		raise arg[0]
	return line_tracer

def global_tracer(frame, event, arg):
	return line_tracer

settrace(global_tracer)
try:
	func()
finally:
	settrace(None)"""
	indented = "\n".join([f"	{line}" for line in code.split("\n")])
	func = "def func():\n" + indented
	exec_env = func + exec_env_boilerplate

	# Trying the execute the generated code
	sio = StringIO()
	with redirect_stdout(sio):
		# We execute the code in a controlled environment
		exec(exec_env, {})

	# Saving the code example with its output
	output = sio.getvalue()
	result = code + "\n# output\n# " + "\n# ".join(output.split("\n")[:-1])
	f.write(result + "\n\n")