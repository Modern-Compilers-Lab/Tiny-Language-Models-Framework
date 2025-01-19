from tqdm import tqdm
from io import StringIO
from contextlib import redirect_stdout

def pytracing_generator(input_file_path,
						output_file_path
						):

	with open(input_file_path, "r") as f:
		data = f.read()
		
	codes = data.split("\n\n")[:-1]

	output_file = open(output_file_path, "w")

	for id_code, code in tqdm(enumerate(codes), total=len(codes)):
		# we only keep the code itslef without the output
		code = (code.split("# output")[0])
		# we prepare the execution env
		indented = "\n".join([f"	{line}" for line in code.split("\n")])
		func = "def func():\n" + indented
		exec_env = func + """
from sys import settrace

code_lines = code.split("\\n")

def line_tracer(frame, event, arg):
	current_step = list(code_lines)
	state_fill = ";".join([f"{key}?{value:}" for key, value in frame.f_locals.items()])
	sio_fill = "&".join(["# " + printed for printed in SIO.getvalue().split("\\n")[:-1]])
	if event == "line":
		current_step[frame.f_lineno - 2] = "@" + current_step[frame.f_lineno - 2] + "$" + state_fill + "|" + sio_fill
		trace.append("#STEP\\n" + "\\n".join(current_step))
	elif event == 'return':
		current_step.append("@^" + "$" + state_fill + "|" + sio_fill)
		trace.append("#STEP\\n" + "\\n".join(current_step))
	return line_tracer

def global_tracer(frame, event, arg):
	return line_tracer

settrace(global_tracer)
try:
	func()
finally:
	settrace(None)"""
		try:
			trace = []
			SIO = StringIO()
			with redirect_stdout(SIO):
				exec(exec_env, {
					"__builtins__":__builtins__,
					"SIO": SIO, 
					"code": code,
					"trace": trace,
					}
				)
			if id_code != len(codes) - 1:
				sep = "\n\n"
			else:
				sep = ""
			output_file.write(code+"\n"+"\n".join(trace)+sep)
		except Exception:
			with open ('error.txt', 'w') as f:
				print("Error in code", id_code)
				# f.write(code)
				# f.write("\n===============================================\n")
				f.write(exec_env)
			break
	output_file.close()

pytracing_generator(input_file_path='30PS_chatGPT_norm_wo_desc.txt',
					output_file_path='traced_30PS_chatGPT_normalized.txt'
					)

# if __name__ == "__main__":
# 	with open("code.txt", "r") as f:
# 		codes = f.read()
# 	codes = codes.split("\n\n")
# 	for code in codes:
# 		code_CoT = str(code)