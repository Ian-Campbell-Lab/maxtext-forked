"""
Copyright 2024 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
     https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=missing-module-docstring, missing-function-docstring

from jax.sharding import Mesh


import unittest
import pytest

import pyconfig


from layers import pipeline
import jax
from jax import numpy as jnp
from jax.sharding import Mesh

import common_types
import pyconfig
import max_utils
from flax.core import meta

import jax.numpy as jnp
from flax import linen as nn
from layers import simple_layer
from train import main as train_main
from train_compile import main as train_compile_main
import os
import shutil

import hashlib

import subprocess

def run_bash_script(script_path, aot_dump_dir, real_dump_dir):
    """Executes a Bash script and returns the completed process object."""
    try:
        result = subprocess.run(
            ["bash", script_path, aot_dump_dir, real_dump_dir],  # Command to run the script
            check=True,             # Raise an exception if the script fails
            stdout=subprocess.PIPE,  # Capture standard output
            stderr=subprocess.PIPE,  # Capture standard error
            text=True               # Decode output and error as text (Python 3.7+)
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")

def find_file_by_substring(directory, substring):
    for filename in os.listdir(directory):
        if substring in filename:
            return os.path.join(directory,filename)
    return None  # Return None if no match found

def delete_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)

def assert_large_files_equal(file_path1, file_path2):
    """Asserts that two potentially large text files have identical content."""

    hasher1 = hashlib.sha256()
    hasher2 = hashlib.sha256()

    with open(file_path1, "rb") as f1, open(file_path2, "rb") as f2:
        # Read files in chunks for memory efficiency
        while True:
            chunk1 = f1.read(8192)  # 8 KB chunks
            chunk2 = f2.read(8192)

            if not chunk1 and not chunk2:  # Reached the end of both files
                break
            hasher1.update(chunk1)
            hasher2.update(chunk2)

    # Handle potential empty files
    if not hasher1.digest() or not hasher2.digest():
        # One or both files are empty
        return False

    if hasher1.hexdigest() != hasher2.hexdigest():
        # Files have different contents
        return False
    return True


hlo_filename_substring="jit_train_step.after_optimizations_after_buffer_assignment.txt"
compile_dump_dir="/tmp/compile_test_xla_dump/aot/"
train_dump_dir="/tmp/compile_test_xla_dump/real/"
delete_dir(compile_dump_dir) # Clean directory before use
delete_dir(train_dump_dir)

run_bash_script("MaxText/test_identical_bash.sh", compile_dump_dir, train_dump_dir)


compile_hlo_file = find_file_by_substring(compile_dump_dir, hlo_filename_substring)
train_hlo_file = find_file_by_substring(train_dump_dir, hlo_filename_substring)
print(f"AOT compiled HLO file: {compile_hlo_file}", flush=True)
print(f"Real runs HLO file: {train_hlo_file}", flush=True)

files_equal = assert_large_files_equal(compile_hlo_file, train_hlo_file)
delete_dir(compile_dump_dir)
delete_dir(train_dump_dir)
assert files_equal, "AOT Compiled and real HLO files are not identical!"
print("AOT Compiled and train HLO files are identical, test passes!")

