import os
import sys
from jinja2 import Environment, FileSystemLoader

# Add the docs directory to the system path
sys.path.insert(0, os.path.abspath('docs'))

# Import the conf.py file as a module
import conf

# Load the template file
file_loader = FileSystemLoader('.')
env = Environment(loader=file_loader)

# Load the template
template = env.get_template('README.template.md')

# Define the context with the metadata from conf.py
context = {
    'description': '',
    'project': conf.project,
    'author': conf.author,
    'release': conf.release
}
# Read the installation instructions from docs/installation.rst
with open('docs/installation.rst', 'r') as file:
    installation_instructions = file.read()

# Add the installation instructions to the context
context['installation'] = installation_instructions
# Render the template with the context
output = template.render(context)

# Write the output to a README.md file
with open('README.md', 'w') as f:
    f.write(output)