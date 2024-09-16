import os
import sys
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader

# Load README template
with open('README.template.md', 'r') as template_file:
    readme_template = template_file.read()

# Define dynamic content
description = "ATPP is a Python library for post-processing FLIR thermal camera data."
installation_instructions = "pip install atpp"
usage_examples = """
```python
from atpp import FlirVideo

# Example usage of ATPP library
flir_video = FlirVideo('path_to_file.ats')
flir_video.process_data()
"""

# Render README.md
template = Template(readme_template) 
readme_content = template.render( description=description, installation_instructions=installation_instructions, usage_examples=usage_examples )


with open('README.md', 'w') as readme_file: readme_file.write(readme_content)

print("README.md has been updated.")