from tqdm import tqdm

def tinypy_code_tracer(input_file_path, output_file_path):
	
	# Print the input and output paths
	print('input_path:', input_file_path)
	print('output_path:', output_file_path)

	# Open the input file containing code snippets
	with open(input_file_path, "r") as f:
		data = f.read()
	
	# Split the data into code snippets using the \n\n seperator
	codes = data.split("\n\n")[:-1]

	# Open the output file for writing traced code snippets
	output_file = open(output_file_path, "w")

	# Open the error file for writing runtime errors
	error_file = open("./data/runtime_error_snippets.txt", "w")
	
	# Initialize a progress bar of traced snippets
	pbar = tqdm(codes, desc="Tracing code snippets | Errors: 0", unit=" Snippet", total=len(codes))
	
	# Initialize a counter for the number of errors
	nb_errors = 0
	
	# Iterate over each code snippet
	for code in pbar:

		# Prepare the execution environment by putting the code snippet inside a function and setting up a Python trace function
		indented = "\n".join([f"	{line}" for line in code.split("\n")])
		func = "def func():\n" + indented
		exec_env = func + """
from sys import settrace

code_lines = code.split("\\n")

def line_tracer(frame, event, arg):
	current_step = list(code_lines)

	# Check runtime value range
	for key, value in frame.f_locals.items():
		if isinstance(value, (int, float)):
			if value < -10 or value > 10:
				raise ValueError(f"Value out of range: {key} = {value}")

	state_fill = ";".join([f"{key}?{value:}" for key, value in frame.f_locals.items()])
	if event == "line":
		current_step[frame.f_lineno - 2] = "@" + current_step[frame.f_lineno - 2] + "$" + state_fill
		trace.append("#STEP\\n" + "\\n".join(current_step))
	elif event == 'return':
		current_step.append("@^" + "$" + state_fill)
		trace.append("#STEP\\n" + "\\n".join(current_step))
	return line_tracer

def global_tracer(frame, event, arg):
	return line_tracer

settrace(global_tracer)
try:
	func()
finally:
	settrace(None)"""
		
		# Launch snippet execution with runtime exceptions handling
		trace = []
		try:
			exec(exec_env, {
				"__builtins__":__builtins__, 
				"code": code,
				"trace": trace,
				}
			)

		# Catch any runtime errors that occur during execution	
		except Exception as e:
			
			# Increment the error counter
			nb_errors += 1

			# Write the error and the code snippet to the error file
			error_file.write(f"Error: {e}\n{code}\n\n")

			# Update the progress bar with the number of errors
			pbar.set_description(f"Tracing code snippets | Errors ({nb_errors})")
			
			# Skip to the next code snippet
			continue
		
		# Write the succesfully traced code snippet to the output file
		output_file.write(code+"\n"+"\n".join(trace)+"\n\n")
	
	# Close the output and error files
	output_file.close()
	error_file.close()


if __name__ == "__main__":
	input_file_path = "./data/snippets.txt"
	output_file_path = "./data/traced_snippets.txt"
	tinypy_code_tracer(input_file_path=input_file_path, output_file_path=output_file_path)