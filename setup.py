from setuptools import setup, find_packages

setup(
    name='brainaudio',
    version='0.1.0',
    description='A project for brain-to-text speech decoding.',
    author='Your Name',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email
    packages=find_packages(),
    # find_packages() automatically discovers and includes all packages
    # (directories with an __init__.py file) in your project.
)