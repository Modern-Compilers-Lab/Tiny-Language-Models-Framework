# Operator Prediction Task
The task requires the model to predict a random operator in the code based on the output

## Usage

### Data Generation
- Data should be generated based on an existing dataset of code snippets with their outputs, as follows:

```bash
python replacer.py --input_file_name input_data.txt --output_file_name output_data.txt
```


- If you want to generate a dataset from scratch, based on old data generation levels, you can run for example:

```bash
python old_generator.py --num_programs 1000 --level ALL --filename output_data.txt --deduplicate
```


### Data Preparation
- Prepare (Tokenize and split) the data by running:

```bash
python prepare.py
```

This should generate the following files: `train.txt`, `test.txt`, `val.bin`, `train.bin`, `test.bin`, `val.bin`, and `meta.pkl`.

## Contact
- **Omar Farouk Zouak**: [omar.zouak@ensia.edu.dz](mailto:omar.zouak@ensia.edu.dz)