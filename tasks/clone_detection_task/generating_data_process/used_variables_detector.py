import ast

class UsedVariablesDetector(ast.NodeVisitor):
    """This class detect the used variables in our simplified code version (variables within the print statments
    or inside the for loop or the if conditions)"""
    def __init__(self):
        self.printed_vars = set()
        self.if_condition_vars = set()

    def visit_Call(self, node):
        # Check if the function being called is 'print'
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            for arg in node.args:
                self._extract_variables(arg, self.printed_vars)
        self.generic_visit(node)

    def visit_If(self, node):
        # Extract variables from the if condition
        self._extract_variables(node.test, self.if_condition_vars)
        self.generic_visit(node)

    def visit_While(self, node):
        # Extract variables from the if condition
        self._extract_variables(node.test, self.if_condition_vars)
        self.generic_visit(node)

    def _extract_variables(self, node, target_set):
        # Recursively extract variable names from the expression
        if isinstance(node, ast.Name):
            target_set.add(node.id)
        elif isinstance(node, ast.BinOp):
            self._extract_variables(node.left, target_set)
            self._extract_variables(node.right, target_set)
        elif isinstance(node, ast.UnaryOp):
            self._extract_variables(node.operand, target_set)
        elif isinstance(node, ast.Call):
            for arg in node.args:
                self._extract_variables(arg, target_set)
        elif isinstance(node, ast.Attribute):
            target_set.add(node.attr)
        elif isinstance(node, ast.Subscript):
            self._extract_variables(node.value, target_set)
        elif isinstance(node, ast.Compare):
            for comparator in node.comparators:
                self._extract_variables(comparator, target_set)
            self._extract_variables(node.left, target_set)
        elif isinstance(node, ast.BoolOp):
            for value in node.values:
                self._extract_variables(value, target_set)

def get_printed_and_condition_variables(code):
    detector = UsedVariablesDetector()
    tree = ast.parse(code)

    detector.visit(tree)

    used_vars = detector.printed_vars | detector.if_condition_vars

    return used_vars