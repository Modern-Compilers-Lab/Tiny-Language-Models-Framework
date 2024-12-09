import argparse
import os
from omegaconf import OmegaConf

action_help = """Action, can be:
- init ; initiliazes the repo according to the venus structure
- sync ; synchronizes the group tags and xp tags from the dbs locally to the confs and to neptune
- ds ; to create a new dataset
- dp ; to create a new dataprep
- xg ; to create a new xperiment group
- xp ; to create a new xperiment
"""

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "venus manager")
	
	parser.add_argument("--action", "-a", default = None, help = action_help)
	parser.add_argument("--tags", "-t", default = None, help = "list of space separated tags for the new artifact")
	parser.add_argument("--datasetid", "-dsi", default = None, help = "id of the dataset to create the dataprep in")
	parser.add_argument("--xpgroupid", "-xgi", default = None, help = "is of the xperiment group to create the xperiment in")
	args = parser.parse_args()
	action = args.action

	# If the action is to initialize the repo as a venus managed repo.
	if action == "init":
		# Creating the .venus folder : will contain special venus files that will be used for the management of the repo.
		os.makedirs(".venus/")
		# Creating the neptune.db file : a yaml database that will contain the neptune project name (the "project" entry)
		# and the neptune api token (the "api-token" entry). This file will be used later by venus when synchronizing the
		# yaml databases (the xperiments.db file and the xpgroups-<xgi>.db files described further) with the neptune associated neptune project
		cfg = OmegaConf.create({
			"project" : None,
			"api-token" : None
		})
		OmegaConf.save(cfg, ".venus/neptune.db")
		# Creating the datasets folder which will contain all the datasets and datapreps of those datasets. Each dataset
		# inside the datasets folder will have a dedicated folder named dataset-<dsi> (the dataset id or "dsi" will be 
		# automatically created by venus). Also each dataset folder will be initialized with a readme.md file + a datapreps-<dsi> 
		# folder which will contain all the datapreps of that dataset. Each dataprep will have its own folder dataprep-<dsi>-<dpi>.
		os.makedirs("datasets/")
		# The datasets folder will contain a datasets.db yaml database file which will store the last-dataset-id and the 
		# tags of the created datasets
		cfg = OmegaConf.create({
			"last-dataset-id" : 0,
			"datasets" : {}
		})
		OmegaConf.save(cfg, "datasets/datasets.db")
		# Creating the xperiments (stylized short form for experiments) folder which will contain all the xpgroups (stylized short form
		# for experiment group). Each xpgroup will have its own folder xpgroup-<xgi> (the experiment group id or "xgi" will be automatically
		# created by venus). Each xpgroup will contain a set experiments each living in a xp-<xgi>-<xpi> subfolder (the experiment id or "xpi" 
		# will be automatically created by venus)
		os.makedirs("xperiments/")
		# The xperiments folder will contain an xperiments.db yaml database file which will store the 
		# last-xpgroup-id and the tags of the created xpgroups
		cfg = OmegaConf.create({
			"last-xpgroup-id" : 0,
			"xpgroups" : {}
		})
		OmegaConf.save(cfg, "xperiments/xperiments.db")
		# Creating the xperiments.db.cache file which will be a snapshot of the xperiments.db file
		# after the last neptune syncing operation. This will be useful to avoid syncronizing unchanged tags with neptune.
		OmegaConf.save(cfg, ".venus/xperiments.db.cache")
	

	# Synchronize the xperiments.db and xpgroup-<xgi>.db tags with xp-<xgi>-<xpi>.conf files + the corresponding neptune runs
	# WARNING: This supposes that all tags modifications are done through the .db files and not directly in the confs nor neptune.
	# ==> synchronizing will overwrite the .conf files and neptune stored tags.
	elif action == "sync":
		# We import neptune
		print("Importing neptune ...")
		import neptune
		# We load the neptune.db file in the .venus folder to get the project name and api-token
		neptune_db = OmegaConf.load(".venus/neptune.db")
		# We laod the xperiments.db and xperiments.db.cache files to check for changes in the xpgroup tags
		xperiments_db = OmegaConf.load("xperiments/xperiments.db")
		xperiments_db_cache = OmegaConf.load(".venus/xperiments.db.cache")
		# For each entry in the xpgroups subdict of xperiments_db
		for xpgroup_id, xpgroup_tags in xperiments_db["xpgroups"].items():
			# We start the syncing operation for xpgroup_id
			print(f"\n######\n\n=== SYNCING {xpgroup_id} ... ===")
			# If the xpgroup is new <==> it's not even present in the xperiments.db.cache => we must sync => we set a 
			# first boolean variable to True. 
			if xpgroup_id not in xperiments_db_cache["xpgroups"]:
				sync_xpgroup_tags = True
			# In case the xpgroup already existed we check if the tags have changed in which case the first boolean variable is set
			# to True and otherwise to False.
			else:
				sync_xpgroup_tags = xpgroup_tags != xperiments_db_cache["xpgroups"][xpgroup_id] 
			# We load the xpgroup-<xgi>.db file and the xpgroup-<xgi>.db.cache files tp check for changes in the xp tags
			xpgroup_db = OmegaConf.load(f"xperiments/{xpgroup_id}/{xpgroup_id}.db")
			xpgroup_db_cache = OmegaConf.load(f".venus/{xpgroup_id}.db.cache")
			# For each entry in the xps subdict of the xpgoup-<xgi>.db file
			for xp_id, xp_tags in xpgroup_db["xps"].items():
				# If the xp is new => we must sync => we set a second boolean variable to True
				if xp_id not in xpgroup_db_cache["xps"]:
					sync_xp_tags = True
				# Else the xp exists so we check for xp tags changes
				else:
					sync_xp_tags = xp_tags != xpgroup_db_cache["xps"][xp_id]
				# If there is a change in eitger the xpgroup tags or the xp tags => we sync with the confs + neptune
				if sync_xpgroup_tags or sync_xp_tags:
					print("-----------------------")
					print(f">>> SYNCING {xp_id} ...")
					xp_conf = OmegaConf.load(f"xperiments/{xpgroup_id}/{xp_id}/{xp_id}.conf")
					xp_conf["xpgroup-tags"] = xpgroup_tags
					xp_conf["xp-tags"] = xp_tags
					OmegaConf.save(xp_conf, f"xperiments/{xpgroup_id}/{xp_id}/{xp_id}.conf")
					
					run = neptune.init_run(
						project = neptune_db["project"],
						custom_run_id = f"XP-{xpgroup_id.split('-')[1]}-{xp_id.split('-')[2]}"
					)
					run["sys/tags"].clear()
					run["sys/tags"].add(xp_tags.split(" "))
					run["sys/group_tags"].clear()
					run["sys/group_tags"].add(xpgroup_tags.split(" "))
					run.stop()
			# Once we finish syncing the xps of the current xpgroup, we update the xpgroup cache
			OmegaConf.save(xpgroup_db, f".venus/{xpgroup_id}.db.cache")
			print("-----------------------")
			print(f"=== ENDED   {xpgroup_id}     ===")
		# Once we finish syncing all the xpgroups, we update the xperiments cache
		OmegaConf.save(xperiments_db, ".venus/xperiments.db.cache")

	## Creating a new dataset
	elif action == "ds":

		# creating the id for the new dataset
		cfg = OmegaConf.load("datasets/datasets.db")
		new_dataset_id = cfg["last-dataset-id"] = cfg["last-dataset-id"] + 1
		cfg["datasets"][f"dataset-{new_dataset_id}"] = args.tags
		OmegaConf.save(cfg, "datasets/datasets.db")

		# creating the base folder boilerplate
		DIR = f"datasets/dataset-{new_dataset_id}/"
		os.makedirs(DIR)
		with open(DIR+f"readme-ds-{new_dataset_id}.md", "w") as f:
			f.write("# DESCRIPTION\n\n")
			f.write("# OBTENTION\n\n")
			f.write("# META-DATA\n\n")
			f.write("# DATA-LOCATION\n\n")
		
		# creating the data subfolder with the .gitignore and .gitkeep files
		os.makedirs(DIR+f"data-ds-{new_dataset_id}/")
		with open(DIR+f"data-ds-{new_dataset_id}/.gitignore", "w") as f:
			f.write("*\n!.gitkeep")
		with open(DIR+f"data-ds-{new_dataset_id}/.gitkeep", "w") as _:
			pass
		
		# creating the datapreps subfolder with the .gitignore, .gitkeep and datapreps.yaml files
		os.makedirs(DIR+f"datapreps-{new_dataset_id}/")
		cfg = OmegaConf.create({
			"last-dataprep-id": 0,
			"datapreps": {}
		})
		OmegaConf.save(cfg, DIR+f"datapreps-{new_dataset_id}/datapreps-{new_dataset_id}.db")


	## Creating a new dataprep within some dataset
	elif action == "dp":

		# getting the datasetid passed as cli argument 
		dsi = int(args.datasetid)

		# creating the new_dataprep_id and its entry in the local datapreps.yaml file
		cfg = OmegaConf.load(f"datasets/dataset-{dsi}/datapreps-{dsi}/datapreps-{dsi}.db")
		new_dataprep_id =  cfg["last-dataprep-id"] = cfg["last-dataprep-id"] + 1
		cfg["datapreps"][f"dataprep-{dsi}-{new_dataprep_id}"] =  args.tags
		OmegaConf.save(cfg, f"datasets/dataset-{dsi}/datapreps-{dsi}/datapreps-{dsi}.db")


		# creating the base folder boilerplate
		DIR = f"datasets/dataset-{dsi}/datapreps-{dsi}/dataprep-{dsi}-{new_dataprep_id}/"
		
		os.makedirs(DIR)
		
		with open(DIR+f"readme-dp-{dsi}-{new_dataprep_id}.md", "w") as f:
			f.write("# DESCRIPTION\n\n")
			f.write("# OBTENTION\n\n")
			f.write("# META-DATA\n\n")
			f.write("# DATA-LOCATION\n\n")
		
		os.makedirs(DIR+f"data-dp-{dsi}-{new_dataprep_id}/")
		
		with open(DIR+f"data-dp-{dsi}-{new_dataprep_id}/.gitignore", "w") as f:
			f.write("*\n!.gitkeep")
		with open(DIR+f"data-dp-{dsi}-{new_dataprep_id}/.gitkeep", "w") as f:
			pass


	## Creating a new xpgroup
	elif action == "xg":

		# creating the id for the new dataset
		cfg = OmegaConf.load("xperiments/xperiments.db")
		new_xpgroup_id = cfg["last-xpgroup-id"] = cfg["last-xpgroup-id"] + 1
		cfg["xpgroups"][f"xpgroup-{new_xpgroup_id}"] = args.tags.strip(" ") + f" XG-{new_xpgroup_id}"
		OmegaConf.save(cfg, "xperiments/xperiments.db")

		# creating the base folder boilerplate
		DIR = f"xperiments/xpgroup-{new_xpgroup_id}/"
		os.makedirs(DIR)

		cfg = OmegaConf.create({
			"last-xp-id": 0,
			"xps": {}
		})
		OmegaConf.save(cfg, DIR+f"xpgroup-{new_xpgroup_id}.db")
		# creating the corresponding cache file (will be used later for syncing)
		OmegaConf.save(cfg, f".venus/xpgroup-{new_xpgroup_id}.db.cache")


	## Creating a new xp
	elif action == "xp":

		# loading the xpgroup_tags
		xgi = int(args.xpgroupid)
		xpgroup_tags = OmegaConf.load("xperiments/xperiments.db")["xpgroups"][f"xpgroup-{xgi}"]

		# creating the new_xp_id and creating the xps-tags entry for the new xp 
		cfg = OmegaConf.load(f"xperiments/xpgroup-{xgi}/xpgroup-{xgi}.db")
		new_xp_id = cfg["last-xp-id"] = cfg["last-xp-id"] + 1
		cfg["xps"][f"xp-{xgi}-{new_xp_id}"] = xp_tags = args.tags
		OmegaConf.save(cfg, f"xperiments/xpgroup-{xgi}/xpgroup-{xgi}.db")
		
		# creating the xp-<id> folder boilerplate
		DIR = f"xperiments/xpgroup-{xgi}/xp-{xgi}-{new_xp_id}/"
		os.makedirs(DIR)
		with open(DIR+f"readme-xp-{xgi}-{new_xp_id}.md", "w") as f:
			f.write("# DESCRIPTION\n\n")
			f.write("# OBTENTION\n\n")
			f.write("# META-DATA\n\n")
			f.write("# MODELS-LOCATION\n\n")
		os.makedirs(DIR+f"train-{xgi}-{new_xp_id}")
		os.makedirs(DIR+f"evals-{xgi}-{new_xp_id}")
		cfg = OmegaConf.create({
			"neptune-id" : f"XP-{xgi}-{new_xp_id}",
			"xpgroup-tags" : xpgroup_tags,
			"xp-tags" : xp_tags
		})
		OmegaConf.save(cfg, DIR+f"xp-{xgi}-{new_xp_id}.conf")


	## Error in the --action cli argument
	else:
		print("ERROR: a valid action (init, ds, dp, xg, xp) must be supplied")
		exit(-1)