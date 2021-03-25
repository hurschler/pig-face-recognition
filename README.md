# Pig Face Recognition using MASK R-CNN, Tensorflow 2.x

[![Python application](https://github.com/hurschler/pig-face-recognition/actions/workflows/python-app.yml/badge.svg)](https://github.com/hurschler/pig-face-recognition/actions/workflows/python-app.yml)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## Run Tensorboard on this project
`tensorboard --logdir logs --bind_all`

http://localhost:8080/index.html

http://localhost:8080/api/getimagejson

http://localhost:8080/api/getimage


## General Git Information
`.gitignore` ignores just files that weren't tracked before.
Run `git reset name_of_file` to unstage the file and keep it.
In case you want to also remove the given file from the repository (after pushing), use `git rm --cached name_of_file`


##Remove .DS_Store hidden in Finder
`find . -name '*.DS_Store' -type f -delete`