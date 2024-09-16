import os
import sys
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader
from pypandoc import convert_file

# Load README template
with open('README.template.md', 'r') as template_file:
    readme_template = template_file.read()

# Define dynamic content
description = "ATPP is a Python library for post-processing FLIR thermal camera data."
installation_instructions = "pip install atpp"
with open('docs/usage.rst', 'r') as usage_file:
    usage_rst = usage_file.read()
    usage_examples = convert_file('docs/usage.rst', 'md')


# Render README.md
template = Template(readme_template) 
readme_content = template.render( description=description, installation_instructions=installation_instructions, usage_examples=usage_examples )


with open('README.md', 'w') as readme_file: readme_file.write(readme_content)

print("README.md has been updated.")