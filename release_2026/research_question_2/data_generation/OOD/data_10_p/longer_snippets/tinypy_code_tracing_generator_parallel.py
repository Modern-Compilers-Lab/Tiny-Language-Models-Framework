from collections import deque, Counter
from tqdm import tqdm
import random
import hashlib
import pickle
import multiprocessing as mp
from multiprocessing import Pool
import argparse
import re 
import os
import ast
import numpy as np
import hashlib
import json
from math import gcd
from functools import reduce




# python3 tinypy_code_tracing_generator_parallel.py  --num-processes 64 --num-snippets 200000000
# python3 tinypy_code_tracing_generator_parallel.py  --num-processes 4 --num-snippets 10000





class Context:
    def __init__(self, init_dict:dict, init_queue:deque):
        self.GID = init_dict
        self.keywords_queue = init_queue
    def enqueue(self, keywords):
        for keyword in keywords:
            self.keywords_queue.append(keyword)
    def dequeue(self):
        if not self.keywords_queue:
            raise IndexError("dequeue from empty queue")
        return self.keywords_queue.popleft()
    def __iter__(self):
        return self
    def __next__(self):
        if not self.keywords_queue:
            raise StopIteration("no more keywords")
        return self.dequeue()

class ContextStack:
    def __init__(self):
        self.context_stack = list()
        self.push()
    def push(self, init_dict=None, init_queue=None):
        
        if init_dict is None:
            init_dict = dict()
        if init_queue is None:
            init_queue = deque()
        
        if not isinstance(init_dict, dict):
            raise TypeError("init_dict must be a dictionary")
        if not isinstance(init_queue, deque):
            raise TypeError("init_queue must be a deque")
        self.context_stack.append(Context(init_dict, init_queue))
    def pop(self):
        if not self.context_stack:
            raise IndexError("pop from empty context stack")
        return self.context_stack.pop()
    def top(self):
        if not self.context_stack:
            raise IndexError("top from empty context stack")
        return self.context_stack[-1]
    def get(self, index):
        if not self.context_stack:
            raise IndexError("get from empty context stack")
        if index < 0 or index >= len(self.context_stack):
            raise IndexError("index out of range")
        return self.context_stack[index]
    def depth(self):
        return len(self) - 1
    def __len__(self):
        return len(self.context_stack)
    def __repr__(self):
        repr_str = "ContextStack(["
        for context in self.context_stack:
            repr_str += f"\n\tContext(\n\t\tGID={context.GID},\n\t\tCKQ={list(context.keywords_queue)},\n\t),"
        repr_str += "\n])"
        return repr_str

class TinyPyGenerator2:
    def __init__(self,arr,cte,max_string_length,nb_workers,worker_id = 0):
        self.NbTeamCount = [0,0]
        self.StrTeamCount = [0,0]
        self.NbSizeDist =  [0,0,0,0,0,0,0,0,0,0]
        self.VarSizeDist = [0,0,0,0,0,0,0,0,0,0]
        self.SnipDepthDist = [0,0,0,0,0,0,0,0,0,0,0]
        self.SnipLengthDist = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.arr = arr
        self.cte = cte
        self.max_string_length = max_string_length
        # self.m = m
        # self.a = a
        # self.c = c 
        self.worker_id = worker_id
        #self.x = 123456789
        #self.series_index = 0
        self.rarity = 1/10
        #print(self.rarity)
        self.nb_workers = nb_workers
        for i in range(self.max_string_length):
            if i == 0:
                continue
            self.next_rand(i,self.arr,self.rarity,step=(self.worker_id%(max(1,int(arr[i][4]*self.rarity)))),cte=self.cte)
        # self.seed_0_01_percent_milstone = 97890657821076 # the (m*rarity)th element in the series , computed by manually running another script (took about 80 minutes)
    def int_to_string(self,x, length):
        s = []
        for _ in range(length):
            s.append(chr(ord('a') + x % 26))
            x //= 26
        return ''.join(reversed(s))

    def resetRand(self,index, arr, cte,worker_id=0):
        arr[index][0] = cte % 26**(index+1)
        arr[index][1] = 0

        x = arr[index][0]
        a = arr[index][2]
        c = arr[index][3]
        m = arr[index][4]
        for _ in range(worker_id):
            x = (a * x + c) % m
        arr[index][1] +=worker_id
        arr[index][0] = x

    def next_rand(self,index, arr, rarity = 1,step=1,cte=None):
        x = arr[index][0]
        a = arr[index][2]
        c = arr[index][3]
        m = arr[index][4]
        if step >= max(1,int(m*rarity)) :
            return x
        for _ in range(step):
            x = (a * x + c) % m
        arr[index][1] +=step
        arr[index][0] = x
        if arr[index][1] >= max(1,int(m*rarity)):
            self.resetRand(index,arr,cte,worker_id=self.worker_id)
            return arr[index][0] 
        return x


    def teamInt(self,x: int) -> int:
        h = hashlib.blake2b(
            x.to_bytes(8, byteorder="little", signed=True),
            digest_size=1
        ).digest()
        return h[0] & 1

    def teamStr(self,s: str) -> int:
        h = hashlib.blake2b(
            s.encode('utf-8'), 
            digest_size=1
        ).digest()
        return h[0] & 1

    # generate a valid number between a and b (inclusive), that is part of team 0
    def generate_number(self,power_limit,teamNb):
        var_length = random.randint(1, power_limit)
        self.NbSizeDist[var_length-1] +=1
        sign = random.choice((-1, 1))
        x = random.randrange(10**(var_length-1)-1, 10**(var_length)-1)
        while self.teamInt(sign*x) != teamNb :
            self.NbTeamCount[1] += 1
            x = random.randrange(10**(var_length-1)-1, 10**(var_length)-1)
        self.NbTeamCount[0] += 1
        return sign*x

    # def generate_string(self,alphabet,reserved,teamNb):
    # 	var_length = random.randint(1, self.max_char_count)
    # 	self.VarSizeDist[var_length-1] +=1
    # 	x = ''.join(random.choices(alphabet, k=var_length))
    # 	while self.teamStr(x) != teamNb or x in reserved : 
    # 		self.StrTeamCount[1] += 1 * (self.teamStr(x) != teamNb)
    # 		x = ''.join(random.choices(alphabet, k=var_length))
    # 	self.StrTeamCount[0] += 1
    # 	return x

    def generate_string(self,alphabet,reserved,teamNb):
        index = random.randint(0, self.max_string_length-1) 
        self.VarSizeDist[index]+=1
        if index == 0 : #case of a single char variable, no limits there return any char from the alphabet
            return random.choice(alphabet)
        else : # apply the rarity rule
            string = self.int_to_string(self.next_rand(index,self.arr,self.rarity,step=(self.nb_workers),cte=self.cte),index+1)
            while string in reserved :
                string = self.int_to_string(self.next_rand(index,self.arr,self.rarity,step=(self.nb_workers),cte=self.cte),index+1)
            return string
    


    def initiate_code(self):
        #=========================================
        # Code instantiation loop
        #=========================================
        while self.context_stack.top().keywords_queue:
            self.keyword = self.context_stack.top().dequeue()

            #|===================================================|
            #|[*] [// User-defined low-level generation rules //]|
            #|===================================================|
            if self.keyword == "ASSIGNMENT":
                lhs = self.generate_string(self.variable_identifiers,self.reserved_words,0)
                u = random.random()
                if u < 0.15:
                    rhs = self.generate_number(self.max_digit_count,0)  
                
                elif u < 0.3:
                    if self.generated_variables:
                        rhs = random.choice(self.generated_variables)
                    else:
                        rhs = self.generate_number(self.max_digit_count,0)  
                
                else:
                    if self.generated_variables:
                        if random.random() < 0.5:
                            op1 = random.choice(self.generated_variables)
                        else:
                            op1 =self.generate_number(self.max_digit_count,0)  
                        
                        if random.random() < 0.5:
                            op2 = random.choice(self.generated_variables)
                        else:
                            op2 =self.generate_number(self.max_digit_count,0)  
                    else:
                        op1 = self.generate_number(self.max_digit_count,0)  
                        op2 = self.generate_number(self.max_digit_count,0)   # it used to be only positive for some reason
                    
                    #op = random.choice(['+', '-'])
                    op = self.main_op
                    rhs = f'{op1} {op} {op2}'

                tabs = '\t' * (self.context_stack.depth())
                self.code_snippet = self.code_snippet + f'{tabs}{lhs} = {rhs}\n'
                self.generated_variables.append(lhs)
                self.context_stack.top().GID["block_line_count"] += 1
                self.line_count += 1


            elif self.keyword == "IF_STATEMENT":
                self.if_count[self.context_stack.depth()]+=1
                if self.generated_variables:
                    if random.random() < 0.5:
                        op1 = random.choice(self.generated_variables)
                    else:
                        op1 = self.generate_number(self.max_digit_count,0)  
                    
                    if random.random() < 0.5:
                        op2 = random.choice(self.generated_variables)
                    else:
                        op2 = self.generate_number(self.max_digit_count,0)  
                else:
                    op1 = self.generate_number(self.max_digit_count,0)  
                    op2 = self.generate_number(self.max_digit_count,0)  
                
                op = random.choice(['<', '>'])
                conditional = f'{op1} {op} {op2}'
                tabs = '\t' * (self.context_stack.depth())
                self.code_snippet = self.code_snippet + f'{tabs}if {conditional}:\n'
                self.context_stack.top().GID["block_line_count"] += 1
                self.line_count += 1
                self.context_stack.push(init_dict={"block_line_count": 0}, init_queue=deque())

            elif self.keyword == "UNINDENT":
                self.context_stack.pop()


            elif self.keyword == 'END':
                pass


            else:
                raise Exception(f'No match for keyword {self.keyword}')

    # SKELETON CONSTRUCTION ALGORITHM
    def construct_skeleton(self):
        self.max_actual_nesting_depth = max(self.max_actual_nesting_depth, self.context_stack.depth())
        #=========================================
        # Keywords sequence initialization
        #=========================================
        keywords_sequence = []
        #|====================================================|
        #|[*] [// User-defined high-level generation rules //]|
        #|====================================================|
        if self.line_count < self.min_line_count:
            if self.context_stack.depth() == 0:
                keywords_sequence.append(random.choice(["ASSIGNMENT", "IF_STATEMENT"]))

            elif self.context_stack.depth() < self.max_nesting_depth:
                if self.context_stack.top().GID.get("block_line_count") != 0:
                    keywords_sequence.append(random.choice(["ASSIGNMENT", "IF_STATEMENT", "UNINDENT"]))

                else:
                    keywords_sequence.append(random.choice(["ASSIGNMENT", "IF_STATEMENT"]))

            else:
                if self.context_stack.top().GID.get("block_line_count") != 0:
                    keywords_sequence.append(random.choice(["ASSIGNMENT", "UNINDENT"]))

                else:
                    keywords_sequence.append("ASSIGNMENT")

        elif self.line_count < self.max_line_count:
            if self.context_stack.depth() == 0:
                keywords_sequence.append(random.choice(["ASSIGNMENT", "IF_STATEMENT", "END"]))
            
            elif self.context_stack.depth() < self.max_nesting_depth:
                if self.context_stack.top().GID.get("block_line_count") != 0:
                    keywords_sequence.append(random.choice(["ASSIGNMENT", "IF_STATEMENT", "UNINDENT", "END"]))
                
                else:
                    new_keyword = random.choice(["ASSIGNMENT", "IF_STATEMENT", "END"])
                    if new_keyword == "END":
                        keywords_sequence.extend(["ASSIGNMENT", "END"])
                    
                    else:
                        keywords_sequence.append(new_keyword)
            
            else:
                if self.context_stack.top().GID.get("block_line_count") != 0:
                    keywords_sequence.append(random.choice(["ASSIGNMENT", "UNINDENT", "END"]))
                
                else:
                    new_keyword = random.choice(["ASSIGNMENT", "END"])
                    if new_keyword == "END":
                        keywords_sequence.extend(["ASSIGNMENT", "END"])
        
                    else:
                        keywords_sequence.append(new_keyword)
        
        else:
            if self.context_stack.depth() != 0 and self.context_stack.top().GID.get("block_line_count") == 0:
                keywords_sequence.extend(["ASSIGNMENT", "END"])
            
            else:
                keywords_sequence.append("END")
        
        #=========================================
        # Top Context Queuing
        #=========================================
        self.context_stack.top().enqueue(keywords_sequence)

    # MAIN ALGORITHM
    def generate_code_snippet(self):

        #=========================================
        # Main global variables
        #=========================================
        self.code_snippet = ""
        self.keyword = "[START]"
        self.context_stack = ContextStack()
        self.if_count = [0,0,0,0,0,0,0,0,0,0,0]
        self.main_op = random.choice(['+', '-'])
        self.max_actual_nesting_depth = 0
        #|==============================================================|
        #|[*] [// User-defined global setup for snippet generation //]  |
        #|==============================================================|
        self.context_stack.top().GID["block_line_count"] = 0
        self.variable_identifiers = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", 
                                     "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                                     "u", "v", "w", "x", "y", "z"]
        self.reserved_words =  [
                        "False", "await", "else", "import", "pass",
                        "None", "break", "except", "in", "raise",
                        "True", "class", "finally", "is", "return",
                        "and", "continue", "for", "lambda", "try",
                        "as", "def", "from", "nonlocal", "while",
                        "assert", "del", "global", "not", "with",
                        "async", "elif", "if", "or", "yield"
                    ]
        self.max_digit_count = 3
        self.max_char_count = 10
        self.max_nesting_depth = 2
        self.line_count = 0
        self.generated_variables = list()
        self.min_line_count = 5
        self.max_line_count = 20


        #=========================================
        # Snippet generation loop
        #=========================================
        while self.keyword != "END":
            self.construct_skeleton()
            self.initiate_code()
        return self.code_snippet


class TinyPyCodeTracer:
    def __init__(self):
        pass


    def trace_snippet(self, snippet):

        #|=======================================================================|
        #|[*] [// User-defined logic for snippet tracing with error handling //] |
        #|=======================================================================|

        # Create the tracing environment
        tracing_env = """
from sys import settrace

def line_tracer(frame, event, arg):
    current_step = code.split("\\n")

    # Check runtime value range
    for key, value in frame.f_locals.items():
        if isinstance(value, (int, float)):
            if value < -999 or value > 999:
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
    settrace(None)
"""
        
        # Insert the code snippet into the tracing environment: put it inside a function named func, and append the tracing environment
        indented_snippet = "\n".join([f"	{line}" for line in snippet.split("\n")])
        snippet_in_tracing_env = "def func():\n" + indented_snippet + tracing_env
        
        # Launch snippet tracing with runtime exceptions handling
        trace = []
        try:
            exec(snippet_in_tracing_env, {
                "__builtins__":__builtins__,
                "code": snippet,
                "trace": trace,
                }
            )
        except Exception as e:
            return snippet, type(e).__name__, str(e)

        # Write the succesfully traced code snippet to the output file
        return snippet+"\n"+"\n".join(trace), None, None

class ParallelTinyPyCodeTracingGenerator:
    def __init__(self, num_snippets):
        # self.percentage = percentage
        self.num_snippets = num_snippets
        self.max_string_length = 5
        self.arr = [[0, 0, 0, 0, 0] for _ in range(self.max_string_length)]
        self.cte = 123456789

        for i in range(1,self.max_string_length):
            m = 26**(i+1)
            factors = self.prime_factors(m)
            a = self.generate_multiplier(m, factors)
            c = 1
            self.arr[i][0]= self.cte % m 
            self.arr[i][1]= 0
            self.arr[i][2]= a
            self.arr[i][3]= c
            self.arr[i][4]= m

        # self.m = (26**11 - 26) // 25   # max space: up to 'zzzzzzzzzz'
        # factors = self.prime_factors(self.m)
        # a_minus_1 = reduce(self.lcm, factors)
        # self.a = a_minus_1 + 1
        # self.c = 1

        # self.test_samples_path = f"/data/aa14546/data/data_2/samples/sampled_combinations_{self.percentage}.pkl"
        # self.test_combinations = ParallelTinyPyCodeTracingGenerator.load_set_pickle(self.test_samples_path)



    def prime_factors(self,n):
        factors = set()
        
        # factor out 2 first
        while n % 2 == 0:
            factors.add(2)
            n //= 2
        
        # check odd numbers from 3 upwards
        i = 3
        while i * i <= n:
            while n % i == 0:
                factors.add(i)
                n //= i
            i += 2
        
        # if remaining n > 1, it's prime
        if n > 1:
            factors.add(n)
        
        return sorted(factors)

    def lcm(self,a, b):
        return a * b // gcd(a, b)

    def lcm_list(self,lst):
        return reduce(self.lcm, lst, 1)

    def generate_multiplier(self,m, prime_factors):
        req = set(prime_factors)
        if m % 4 == 0:
            req.add(4)

        L = self.lcm_list(req)
        a = L + 1

        if a >= m:
            raise ValueError("No valid multiplier < m")

        return a
    # def prime_factors(self,n):
    # 	factors = set()
        
    # 	# factor out 2 first
    # 	while n % 2 == 0:
    # 		factors.add(2)
    # 		n //= 2
        
    # 	# check odd numbers from 3 upwards
    # 	i = 3
    # 	while i * i <= n:
    # 		while n % i == 0:
    # 			factors.add(i)
    # 			n //= i
    # 		i += 2
        
    # 	# if remaining n > 1, it's prime
    # 	if n > 1:
    # 		factors.add(n)
        
    # 	return sorted(factors)



    # def lcm(self,x, y):
    # 	return x * y // gcd(x, y)


    def worker_generate_batch(self, args):
        
        worker_id, batch_size, seed_offset, max_consecutive_duplicates,num_processes = args
        # Set unique seed for each worker
        random.seed(454647 + seed_offset + worker_id)
        
        # Initialize worker-specific generator and tracer
        worker_tpg2 = TinyPyGenerator2(self.arr,self.cte,self.max_string_length,num_processes,worker_id)
        worker_tpct = TinyPyCodeTracer()
        
        # Local tracking
        local_hashes = set()
        consecutive_duplicates = 0
        results = {
            'snippets': [],
            'hashes': set(),
            'stats': {
                'generated': 0,
                'duplicates': 0,
                'value_errors': 0,
                'unbound_errors': 0,
                'other_errors': 0,
                'NumberTeamDist':[0,0],
                'StringTeamDist':[0,0],
                'NumberSizeDist':[0,0,0,0,0,0,0,0,0,0],
                'StringSizeDist':[0,0,0,0,0,0,0,0,0,0],
                'SnippetDepthDist':[0,0,0,0,0,0,0,0,0,0,0],
                'SnippetLengthDist':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            }
        }


        
        generated_count = 0
        
        while generated_count < batch_size and consecutive_duplicates < max_consecutive_duplicates:
            try:
                # Generate code snippet
                code_snippet = worker_tpg2.generate_code_snippet().strip("\n")
                
                code_hash = hashlib.sha256(code_snippet.encode('utf-8')).hexdigest()

                # Check for local duplicates
                if code_hash in local_hashes:
                    consecutive_duplicates += 1
                    results['stats']['duplicates'] += 1

                # Reset consecutive duplicates counter
                consecutive_duplicates = 0
                local_hashes.add(code_hash)

                # Trace the snippet
                exec_trace, error_type, error_msg = worker_tpct.trace_snippet(code_snippet)

                if error_type is not None:
                    if error_type == "ValueError":
                        results['stats']['value_errors'] += 1
                    elif error_type == "UnboundLocalError":
                        results['stats']['unbound_errors'] += 1
                    else:
                        results['stats']['other_errors'] += 1
                        # Log unexpected errors
                        print(code_snippet)
                        print(f"Worker {worker_id}: Unexpected error {error_type}: {error_msg}")
                else:
                    if worker_tpg2.line_count<=worker_tpg2.max_line_count and worker_tpg2.line_count>10 and worker_tpg2.if_count[0] <=2 and worker_tpg2.if_count[1] <=2 and worker_tpg2.if_count[2] <=2 and worker_tpg2.if_count[3] <=2 and worker_tpg2.if_count[4] <=2 and worker_tpg2.if_count[5] <=2 and worker_tpg2.if_count[6] <=2 and worker_tpg2.if_count[7] <=2 and worker_tpg2.if_count[8] <=2 and worker_tpg2.if_count[9] <=2 and worker_tpg2.if_count[10] <=2:
                        # one last verification step to get 
                        # Successfully generated snippet
                        worker_tpg2.SnipDepthDist[worker_tpg2.max_actual_nesting_depth] +=1
                        worker_tpg2.SnipLengthDist[worker_tpg2.line_count] +=1
                        results['snippets'].append(exec_trace)
                        results['hashes'].add(code_hash)
                        generated_count += 1
                        results['stats']['generated'] += 1


            except Exception as e:
                print(f"Worker {worker_id}: Exception in generation: {e}")
                continue

        results['stats']['NumberTeamDist']= worker_tpg2.NbTeamCount
        results['stats']['StringTeamDist']= worker_tpg2.StrTeamCount
        results['stats']['NumberSizeDist']= worker_tpg2.NbSizeDist
        results['stats']['StringSizeDist']= worker_tpg2.VarSizeDist
        results['stats']['SnippetDepthDist']= worker_tpg2.SnipDepthDist
        results['stats']['SnippetLengthDist']= worker_tpg2.SnipLengthDist

        return results
    
    def merge_and_deduplicate(self, all_results, global_hashes):
        """Merge results from all workers and remove global duplicates"""
        snippets = []
        total_stats = {
            'generated': 0,
            'duplicates': 0, 
            'value_errors': 0,
            'unbound_errors': 0,
            'other_errors': 0,
            'global_duplicates': 0,
            'NumberTeamDist':  [0,0],
            'StringTeamDist':  [0,0],
            'NumberSizeDist':  [0,0,0,0,0,0,0,0,0,0],
            'StringSizeDist':  [0,0,0,0,0,0,0,0,0,0],
            'SnippetDepthDist':[0,0,0,0,0,0,0,0,0,0,0],
            'SnippetLengthDist':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        }
        
        for result in all_results:
            # Merge train stats
            for key in ['generated', 'duplicates', 'value_errors', 'unbound_errors', 'other_errors']:
                total_stats[key] += result['stats'].get(key, 0)

            for key in ['NumberTeamDist', 'StringTeamDist', 'NumberSizeDist', 'StringSizeDist', 'SnippetDepthDist','SnippetLengthDist']:
                temp = [x + y for x, y in zip(result['stats'].get(key,0), total_stats[key])]
                total_stats[key] = temp
            # Check for global duplicates and add unique snippets
            for snippet in result['snippets']:
                snippet_hash = hashlib.sha256(snippet.encode('utf-8')).hexdigest()
                if snippet_hash not in global_hashes:
                    global_hashes.add(snippet_hash)
                    snippets.append(snippet)
                else:
                    total_stats['global_duplicates'] += 1


        return snippets, total_stats

    def create_corpus(self, num_processes=None):
        """Main corpus creation function with parallelization"""
        if num_processes is None:
            num_processes = mp.cpu_count()

        print(f"Starting parallel generation with {num_processes} processes")

        # Configuration 
        target_programs = self.num_snippets  # Total snippets to generate
        batch_size = 10000  # Snippets per batch per worker 
        max_consecutive_duplicates = 50  # Stop worker if too many consecutive duplicates 
        
        # Global state 
        global_hashes = set() 
        total_generated = 0 
        
        # Simple counters for final stats
        total_stats = {'generated': 0, 'duplicates': 0, 'value_errors': 0, 'unbound_errors': 0, 'other_errors': 0, 'global_duplicates': 0,
            'NumberTeamDist':  [0,0],
            'StringTeamDist':  [0,0],
            'NumberSizeDist':  [0,0,0,0,0,0,0,0,0,0],
            'StringSizeDist':  [0,0,0,0,0,0,0,0,0,0],
            'SnippetDepthDist':[0,0,0,0,0,0,0,0,0,0,0],
            'SnippetLengthDist':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
        os.makedirs("./data", exist_ok=True)
        # Output files
        output = open(f"./data/raw_ID.txt", "w") 
        
        # Progress bar - ONLY for train snippets
        pbar = tqdm(total=target_programs, desc="Train snippets", position=0, leave=True) 
        
        try: 
            with Pool(processes=num_processes) as pool: 
                batch_num = 0 
                
                while total_generated < target_programs: 
                    # Prepare worker arguments 
                    remaining = target_programs - total_generated 
                    current_batch_size = min(batch_size, remaining // num_processes + 1) 
                    
                    worker_args = [
                        (worker_id, current_batch_size, batch_num * num_processes, max_consecutive_duplicates,num_processes) 
                        for worker_id in range(num_processes) 
                    ] 
                    
                    batch_results = pool.map(self.worker_generate_batch, worker_args) 
                    
                    # Merge and deduplicate results 
                    snippets, batch_stats = self.merge_and_deduplicate( 
                        batch_results, global_hashes 
                    ) 
                    
                    # Update stats - simple addition
                    for key in ['generated', 'duplicates', 'value_errors', 'unbound_errors', 'other_errors','global_duplicates']:
                        total_stats[key] += batch_stats[key]

                    for key in ['NumberTeamDist', 'StringTeamDist', 'NumberSizeDist', 'StringSizeDist', 'SnippetDepthDist','SnippetLengthDist']:
                        temp = [x + y for x, y in zip(batch_stats.get(key,0), total_stats[key])]
                        total_stats[key] = temp

                    # Write to files 
                    for snippet in snippets: 
                        output.write(snippet + "\n\n") 
                                    
                    # Update
                    new_generated = len(snippets) 
                    total_generated += new_generated 
                    pbar.update(new_generated) 
                    
                    batch_num += 1 
                    
        finally: 
            output.close() 
            pbar.close() 
            
        # Final stats - simple and clear
        print(f"\n=== FINAL RESULTS ===")
        print(f"TRAIN SNIPPETS:")
        print(f"  Generated: {total_stats['generated']}")
        print(f"  Total duplicates found: {total_stats['global_duplicates']}")
        print(f"  Value errors: {total_stats['value_errors']}")
        print(f"  Unbound errors: {total_stats['unbound_errors']}")
        print(f"  Other errors: {total_stats['other_errors']}")

        print(f"  NumberTeamDist: {total_stats['NumberTeamDist']}")
        print(f"  StringTeamDist: {total_stats['StringTeamDist']}")
        print(f"  NumberSizeDist: {total_stats['NumberSizeDist']}")
        print(f"  StringSizeDist: {total_stats['StringSizeDist']}")
        print(f"  SnippetDepthDist: {total_stats['SnippetDepthDist']}")
        print(f"  SnippetLengthDist: {total_stats['SnippetLengthDist']}")
        with open("./data/stats.txt", "w") as f:
            json.dump(total_stats,f,indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ParallelTinyPyCodeTracingGenerator with custom number of processes and snippets to generate")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=8,
        help="Number of processes to use (default: 8)"
    )
    parser.add_argument(
        "--num-snippets",
        type=int,
        default=100,
        help="Number of snippets to generate (default: 100)"
    )
    args = parser.parse_args()
    parallel_generator = ParallelTinyPyCodeTracingGenerator(num_snippets=args.num_snippets)
    parallel_generator.create_corpus(num_processes=args.num_processes)

    # import matplotlib.pyplot as plt

    # Read all generated snippets
    with open("./data/raw_ID.txt", "r") as f:
        all_code = f.read()

    # Split by snippet
    snippets = all_code.strip().split("\n\n")

    all_strings_len4 = []

    for snippet in snippets:
        # Extract variable-like strings of length 4 in this snippet
        strings_len4 = re.findall(r'\b[a-z]{1}\b', snippet)
        # Deduplicate within snippet
        strings_len4 = set(strings_len4)
        all_strings_len4.extend(strings_len4)

    # Count occurrences across snippets
    counter_len4 = Counter(all_strings_len4)


    # # Top 10 most common strings
    # top_strings = counter_len4.most_common(len(counter_len4))
    print(len(counter_len4))
    # labels, counts = zip(*top_strings)

    # # Plot and save
    # plt.figure(figsize=(10,6))
    # plt.bar(labels, counts, color='skyblue')
    # plt.title("Top 10 Most Common Strings of Length 4")
    # plt.xlabel("Strings")
    # plt.ylabel("Occurrences (per-snippet deduped)")
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig("./data/strings_len4_distribution.png")
    # plt.close()
