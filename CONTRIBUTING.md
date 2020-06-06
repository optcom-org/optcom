# Contributing

## How can I contribute ?

Any contribution for Optcom is welcome and there are tasks for
contributors of all experience levels.

You can contribute to Optcom by coding your own component and share it
with the community by creating a pull request.

For more advanced contribution, please see the [`ROADMAP.md`](ROADMAP.md)
for available tasks. [Existing issues](https://github.com/optcom-org/optcom/issues)
can also be solved. Help in testing and documenting as well as new idea
or suggestion are always welcome as well !

## Quick start

In order to have the latest available code, clone it form the git
repository:
```sh
git clone https://github.com/optcom-org/optcom.git
```
You can install locally the downloaded version of Optcom. To do so, go
in the directory where the root directory is, and run:
```sh
python3 -m pip install -e optcom
```
Dependencies should have been installed. If any problem is faced with
dependencies, it can be installed manually:
```sh
python3 -m pip install -r requirements.txt
```
You have now the latest code of Optcom and you are ready to code !


## Pull Request Checklist

In order to submit your pull request, followed the guidelines;

- Read the [contributing guidelines](CONTRIBUTING.md).
- Make sure you are familiar with [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)
- Run the [tests](https://github.com/optcom-org/optcom/blob/master/CONTRIBUTING.md#test)
- Run [mypy](https://github.com/optcom-org/optcom/blob/master/CONTRIBUTING.md#type-hints)
- Run the [doc](https://github.com/optcom-org/optcom/blob/master/CONTRIBUTING.md#documentation)

## Test

Test in Optcom is executed by chance of [Pytest](https://docs.pytest.org/en/latest/)
and are located in [`tests/`](tests/). Before pulling, write tests for
your code and make sure that all tests are successful. To do so, first
make sure that Pytest is installed:
```sh
python3 -m pip install pytest
```
Then go in the root directory and run:
```sh
pytest tests/
```

## Type hints

Optcom uses type hints, an emulation of static typing for Python.
This is believed to be a great way to avoid a common source of errors
created by dynamic typing in Python. Support for type hints can be found
[here](https://docs.python.org/3/library/typing.html). Before
pulling, make sure than your type hints are consistent by chance of
[Mypy](http://mypy-lang.org/). Go in the root directory and run:
```sh
mypy --ignore-missing-imports optcom
```

## Coding Style

Coding style must comply with the following conventions:

- General style guide must comply with [PEP 8](https://www.python.org/dev/peps/pep-0008/)

- Type hints must comply with [PEP 484](https://www.python.org/dev/peps/pep-0484/) and [PEP 256](https://www.python.org/dev/peps/pep-0526/)

- Docstrings must comply with [PEP 257](https://www.python.org/dev/peps/pep-0257/)
and be written in [Numpy Style](https://numpydoc.readthedocs.io/en/latest/format.html)

There are a few conventions for Optcom which are exceptions of the
aforementioned code styles:

- A second line on a conditional statement with a tab is allowed.
- A line which begins by ``# =``, followed by a number of ``=`` signs to
reach 79 characters, separates methods in class.
- A line which begins by ``# -``, followed by a number of ``-`` signs to
reach 79 characters, separates overload methods and property methods in
class.

A few clarifications:
- String can be divided on multiple lines and put in parentheses
- A blank line must be put before the return statement in function.

Moreover a header text must be put at the beginning of every file, find
it in [`optcom_header.txt`](optcom_header.txt) . Do not forget to put
your name as author (cf. last line).

It is advised to work with a text editor that can provide you with
vertical line marks and put one at 72 characters for docstrings and
one at 79 characters for code.

One last thing, there is probably no need to say so, but your code
should come with comments !

## Documention

Please put docstrings anywhere you judge it necessary. Before
pulling, make sure that the documentation compile. To do so, make sure
that you have sphinx and sphinx-apidoc installed:
```sh
python3 -m pip install sphinx
```
Then go in the root directory and create automatic documentation files
by chance of apidoc:
```sh
sphinx-apidoc -o docs/source/ optcom
```
Once this is done, build the doc. Go first in [`docs/`](docs/)
and type:
```sh
make clean && make html
```
If no errors occur, you should be able to see the result in
[`docs/build/`](docs/build/)
