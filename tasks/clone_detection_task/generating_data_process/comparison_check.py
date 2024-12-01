import ast

class ComparisonDetector(ast.NodeVisitor):
    """This class check if our code snippet contain a comparison between two diffrent variables
    or if we have a variable depanding on 2 other variables
    example:
    a = b - c"""
    def __init__(self):
        self.has_comparison_between_diff_vars = False
        self.has_binary_operation_between_diff_vars = False

    def visit_Compare(self, node):
        # Check if the left side of the comparison is a variable
        if isinstance(node.left, ast.Name):
            left_var = node.left.id

            # Check all comparators (right-hand side of the comparison)
            for comparator in node.comparators:
                if isinstance(comparator, ast.Name) and comparator.id != left_var:
                    # If there is a comparison between two different variables
                    self.has_comparison_between_diff_vars = True
                    return  # No need to continue if we already found a match

        self.generic_visit(node)

    def visit_BinOp(self, node):
        # check if the binary operation is within a print statement
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.FloorDiv)):
            if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name):
                if node.left.id != node.right.id:
                    self.has_binary_operation_between_diff_vars = True
                    return

        self.generic_visit(node)

    def visit_Call(self, node):

        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            # visit the arguments of the print statement to check for binary operations
            for arg in node.args:
                self.visit(arg)

        self.generic_visit(node)

def has_diff_var_comparison(code):
    tree = ast.parse(code)
    detector = ComparisonDetector()
    detector.visit(tree)
    return detector.has_comparison_between_diff_vars or detector.has_binary_operation_between_diff_vars
