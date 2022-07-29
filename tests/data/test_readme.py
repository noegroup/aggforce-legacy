
import os


def test_readme():
    # change to project root directory
    project_root = os.path.join(os.path.dirname(__file__), "../../")
    cwd = os.getcwd()
    os.chdir(project_root)

    # parse the "```python" block from the README file
    code = []
    with open("README.md", "r") as readme_file:
        in_python_block = False
        for line in readme_file:
            if "```" in line:
                in_python_block = False
            if "```python" in line:
                in_python_block = True
                continue
            if in_python_block:
                code.append(line)

    # execute the code
    exec("".join(code))

    # undo the directory change
    os.chdir(cwd)
