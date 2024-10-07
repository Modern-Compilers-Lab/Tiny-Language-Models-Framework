import argparse
import os
import glob

action_help = """Action, can be:
1- ds ; to create a new dataset
2- dp ; to create a new dataprep
3- xg ; to create a new xperiment group
3- xp ; to create a new xperiment 
"""

def generate_tags_string(tags):
	if tags is not None:
		tags_string = "-"+"-".join(tags.split(" "))
	else:
		tags_string = ""
	return tags_string

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "venus manager")
	
	parser.add_argument("--action", "-a", default = None, help = action_help)
	parser.add_argument("--tags", "-t", default = None, help = "list of space separated tags for the new artifact")
	parser.add_argument("--datasetid", "-dsi", default = None, help = "id of the dataset to create the dataprep in")
	parser.add_argument("--xpgroupid", "-xgi", default = None, help = "is of the xperiment group to create the xperiment in")

	args = parser.parse_args()
	action = args.action
	

	## Creating a new dataset
	if action == "ds":

		# creating the id for the new dataset
		max_id = -float("inf")
		for dataset in glob.glob("datasets/*"):
			if max_id < (id := int(dataset.split("/")[-1].split("-")[1])):
				max_id = id
		new_id = max_id + 1
		
		# creating the tags_string
		tags_string = generate_tags_string(args.tags)

		# creating the base folder boilerplate
		DIR = f"datasets/dataset-{new_id}{tags_string}/"
		os.makedirs(DIR)
		with open(DIR+".readme.md", "w") as f:
			f.write("# DESCRIPTION\n\n")
			f.write("# OBTENTION\n\n")
			f.write("# META-DATA\n\n")
			f.write("# DATA-LOCATION\n\n")
		os.makedirs(DIR+"data")
		os.makedirs(DIR+"datapreps")
	
	
	## Creating a new dataprep within some dataset
	elif action == "dp":

		# checking if the user provided the datasetid
		if args.datasetid is None:
			print("ERROR: dsi (dataset id) must be passed as argument to create a new dp (dataprep)")
			exit(-1)
		
		# add a try catch block here in case the user passes something that isn't an integer
		dsi = int(args.datasetid)
		
		# resolving the dataset in which we want to create the new dataprep
		resolved = False
		for dataset in glob.glob("datasets/*"):
			if dsi == int(dataset.split("/")[-1].split("-")[1]):
				resolved = True
				break
		if not resolved:
			print("ERROR: the supplied dsi (dastaset id) doesn't exist")
			exit(-1)

		# creating the id for the new dataprep
		max_id = -float("inf")
		for dataprep in glob.glob(dataset+"/datapreps/*"):
			print(dataprep)
			if max_id < (id := int(dataprep.split("/")[-1].split("-")[1])):
				max_id = id
		new_id = max_id + 1

		# creating the tags_string
		tags_string = generate_tags_string(args.tags)

		# creating the base folder boilerplate
		DIR = dataset+f"/datapreps/dataprep-{new_id}{tags_string}/"
		os.makedirs(DIR)
		with open(DIR+".readme.md", "w") as f:
			f.write("# DESCRIPTION\n\n")
			f.write("# OBTENTION\n\n")
			f.write("# META-DATA\n\n")
			f.write("# DATA-LOCATION\n\n")
		os.makedirs(DIR+"data")
	

	## Creating a new xpgroup
	elif action == "xg":

		# creating the id for the new xpgroup
		max_id = -float("inf")
		for xpgroup in glob.glob("xperiments/*"):
			if max_id < (id := int(xpgroup.split("/")[-1].split("-")[1])):
				max_id = id
		if max_id == -float("inf"): max_id = 0
		new_id = max_id + 1
		
		# creating the tags_string
		tags_string = generate_tags_string(args.tags)
		
		# creating the base folder boilerplate
		DIR = f"xperiments/xpgroup-{new_id}{tags_string}/"
		os.makedirs(DIR)
		with open(DIR+".readme.md", "w") as f:
			f.write("# DESCRIPTION\n\n")
	
	
	## Crearting a new xp
	elif action == "xp":

		# checking if the user provided the xpgroupid
		if args.xpgroupid is None:
			print("ERROR: xgi (xpgroup id) must be passed as argument to create a new dp (dataprep)")
			exit(-1)
		
		# add a try catch block here in case the user passes something that isn't an integer
		xgi = int(args.xpgroupid)
		
		# resolving the xpgroup on which we want to create the new xp
		resolved = False
		for xpgroup in glob.glob("xperiments/*"):
			if xgi == int(xpgroup.split("/")[-1].split("-")[1]):
				resolved = True
				break
		if not resolved:
			print("ERROR: the supplied xgi (xpgroup id) doesn't exist")
			exit(-1)

		# creating the id for the new xp
		max_id = -float("inf")
		for xp in glob.glob(xpgroup+"/*"):
			if max_id < (id := int(xp.split("/")[-1].split("-")[1])):
				max_id = id
		if max_id == -float("inf"): max_id = 0
		new_id = max_id + 1
		
		# creating the tags_string
		tags_string = generate_tags_string(args.tags)

		# creating the base directory boilerplate
		DIR = xpgroup + f"/xp-{new_id}{tags_string}/"
		os.makedirs(DIR)
		with open(DIR+".readme.md", "w") as f:
			f.write("# DESCRIPTION\n\n")
			f.write("# OBTENTION\n\n")
			f.write("# META-DATA\n\n")
			f.write("# MODELS-LOCATION\n\n")
		os.makedirs(DIR+"train")
		os.makedirs(DIR+"evals")
	
	else:
		print("ERROR: a valid action (ds, dp, xg, xp) must be supplied")
		exit(-1)