import datetime
import os
import hashlib
from tqdm import tqdm
from io import StringIO
from contextlib import redirect_stdout

def pytracing_generator(input_file_path=None,
						include_line_numbers_in_program=None,
						include_program_output=None,
						include_line_number_in_trace=True,
						include_program_line_in_trace=True,
						output_file_path="./tracing.txt"
						):

	with open(input_file_path, "r") as f:
		data = f.read()
		
	code_examples = data.split("\n\n")
	code_examples = code_examples[:1]

	output_file = open(output_file_path, "w")

	for program in tqdm(code_examples):
		program = (program.split("# output")[0])[:-1]
		trace = []
		indented = "\n".join([f"	{line}" for line in program.split("\n")])
		func = "def func():\n" + indented
		exec_env = func + """
from sys import settrace

last_state = None
program_lines = program.split("\\n")

def line_tracer(frame, event, arg):
    global last_state
    state_fill=",".join([f"{key}={value}" for key, value in frame.f_locals.items()])
    if event == "line":
        if include_line_number_in_trace and include_program_line_in_trace:
            line_fill = lnp + str(frame.f_lineno-1) + lns + program_lines[frame.f_lineno-2]
        elif include_line_number_in_trace:
            line_fill = lnp + str(frame.f_lineno-1) + lns
        elif include_program_line_in_trace:
            line_fill = program_lines[frame.f_lineno-2]
        else:
            line_fill = ""
    elif event == "exception":
        line_fill = "0"
    else: # i.e. event == "return"
        line_fill = "-1"
    current_state = f"\\n# newLine: {line_fill}\\n# newVars: {state_fill}"
    trace.append(display_program + "\\n# current" + str(last_state) + "\\n# new" + current_state)
    last_state = f"\\n# currentLine: {line_fill}\\n# currentVars: {state_fill}"
    return line_tracer

def global_tracer(frame, event, arg):
    return line_tracer

settrace(global_tracer)
try:
    func()
except ZeroDivisionError:
    trace.pop()
settrace(None)"""
		if include_line_numbers_in_program:
			lnp = include_line_numbers_in_program["ln_prefix"]
			lns = include_line_numbers_in_program["ln_suffix"]
			display_program = "\n".join([f"{lnp}{i+1}{lns}{line}" for i, line in enumerate(program.split("\n"))])
		else:
			display_program = program
		try:
			SIO = StringIO()
			with redirect_stdout(SIO):
				exec(exec_env, {
					"__builtins__":__builtins__,
					"include_output":include_program_output=="within",
					"SIO": SIO, 
					"include_line_number_in_trace": include_line_number_in_trace,
					"include_program_line_in_trace": include_program_line_in_trace,
					"program": program,
					"lnp": include_line_numbers_in_program["ln_prefix"] if include_line_numbers_in_program != None else "",
					"lns": include_line_numbers_in_program["ln_suffix"] if include_line_numbers_in_program != None else "",
					"trace": trace,
					"display_program":display_program
					}
				)
		except Exception:
			pass
		trace = trace[1:]
		output_file.write("\n\n".join(trace)+"\n\n")
	output_file.close()

include_line_numbers_in_program = {"ln_prefix":"@", "ln_suffix":"; "}
pytracing_generator("./data.txt",
					include_line_numbers_in_program = include_line_numbers_in_program,
					include_line_number_in_trace=True,
					include_program_line_in_trace=True
					)