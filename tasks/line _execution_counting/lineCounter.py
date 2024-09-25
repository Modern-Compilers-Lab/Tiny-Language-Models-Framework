import sys
from io import StringIO
from contextlib import redirect_stdout

def line_counter(code_snippet):
    """
    this function counts how many lines of code in total have been executed 
    the function follows the following rules :
        - a line is not counted if :
            - it falls in a condition bloc where the condition is not verified
            - it falls in a loop where the number of iterations is equal to zero
        - a line is counted as much as it has been iterated through "if it sits in a for loop bloc for example "
    """
    counter = 0
    
    def trace_lines(frame, event, arg):
        nonlocal counter # declaring the outer variable
        if event == 'line': # every time the tracer detects the execution of a line of code
            filename = frame.f_code.co_filename
            if filename == '<string>' : # counting only the lines that are in the code snippet we provided "and not in some other internal libraries"
                counter += 1 # increment the global counter
        return trace_lines
    

    # Set the trace function
    sys.settrace(trace_lines)

    # Capture the output of the program.
    SIO = StringIO()
    with redirect_stdout(SIO):
        # executing the code, the execution is being traced by the trace_lines() function that has been set previously
        exec(code_snippet,{'__file__': '<string>'}) # Execute the code and setting the "fake file" name to <string> so that we can recognise this code snippet later in trace_lines()

    # Disable the trace function
    sys.settrace(None)

    return counter

code_snippet = """e = 6
e = 0
e = 7
if not (e != e) or ( e <= e) :
	print(e)
else :
	print(e)"""
number = line_counter(code_snippet)
print(f"\n{number} lines executed successfully\n")