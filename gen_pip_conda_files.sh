#!/usr/bin/env bash

echo "generating pip requirements file"
poetry export --without-hashes -o requirements.txt
