# Contribution guide

Let's look at the possible ways to contribute to the project:
- Voting for suggestions
  - Vote for the suggestion you are interested in on [GitHub Discussions](https://github.com/etna-team/etna/discussions/categories/improvements)
- Taking part in discussions
  - Take part in discussions on [GitHub Discussions](https://github.com/etna-team/etna/discussions)
- Improving documentation
  - Create an [issue about improving documentation](https://github.com/etna-team/etna/issues/new/choose)
- Sending bug report
  - Create an [issue about bug](https://github.com/etna-team/etna/issues/new/choose)
- Suggesting an idea
  - If there is suggestion with very similar idea on [GitHub Discussions](https://github.com/etna-team/etna/discussions/categories/improvements), vote for it
  - Otherwise, add new idea to [GitHub Discussions](https://github.com/etna-team/etna/discussions/categories/improvements)
  - You could continue this step by taking part in discussion
- Pointing out a problem
  - If there is suggestion with very similar problem on [GitHub Discussions](https://github.com/etna-team/etna/discussions/categories/improvements), vote for it
  - Otherwise, add new problem to [GitHub Discussions](https://github.com/etna-team/etna/discussions/categories/improvements)
  - You could continue this step by taking part in discussion
- Making a pull request
  - If there is a feature you want to add or bug you want to fix
  - Follow a [step-by-step guide](##step-by-step-guide-for-making-a-pull-request)

## Step-by-step guide for making a pull request

Every good PR usually consists of:
- feature implementation :)
- documentation to describe this feature to other people
- tests to ensure everything is implemented correctly.

### 1. Before the PR
Please ensure that you have read the following docs:
- [documentation](https://docs.etna.ai/stable/),
- [tutorials](https://github.com/etna-team/etna/tree/master/examples),
- [changelog](https://github.com/etna-team/etna/blob/master/CHANGELOG.md).

### 2. Setting up your development environment

Before writing any code it is useful to set up a development environment.
1. Clone etna library to some folder and go inside:
```bash
git clone https://github.com/etna-team/etna.git etna/
cd etna
```
2. Run installation with `poetry` ([poetry installation guide](https://python-poetry.org/docs/#installation)):
```bash
poetry install -E all-dev
```
3. Activate virtual environment created by poetry:
```bash
poetry shell
```

To connect virtual environment interpreter to IDE the `which python` command can be useful.

### 3. Choosing a task

Ready to do tasks are present at [GitHub Issues](https://github.com/etna-team/etna/issues):
- Pick an issue with status "Todo" on a [board](https://github.com/orgs/etna-team/projects/1);
- Pay attention for the label "good first issue" if you are new to the project.

If there aren't interesting tasks go to [GitHub Discussions with improvements](https://github.com/etna-team/etna/discussions/categories/improvements):
1. Pick an improvement you want to work with;
2. Leave a comment in the discussion that you want to work on this;
3. Take part in discussion about the implementation details;
4. Wait for the issue to be created based on this discussion.

After you picked your issue to work with:
1. Leave a comment in the issue that you want to work on this task;
2. If you need more context on a specific issue, please ask, and we will discuss the details.

### 4. Doing a task

You can also join our [ETNA Community telegram chat](https://t.me/etna_support) to make it easier to discuss.
Once you finish implementing a feature or bugfix, please send a Pull Request.

If you are not familiar with creating a Pull Request, here are some guides:
- [Creating a pull request](https://help.github.com/articles/creating-a-pull-request/);
- [Creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

### 5. Writing tests

Do not forget to check that your code passes the unit tests.
```bash
pytest tests -v
pytest etna -v --doctest-modules
```

ETNA uses the [`black`](https://github.com/psf/black) and [`flake8`](https://github.com/pycqa/flake8) with several plugins 
for coding style checks as well as [`mypy`](https://github.com/python/mypy) for type checks, and you must ensure that your code follows it. 
```bash
make format
```

If any of checks fails, the CI will fail and your Pull Request won't be merged.

### 6. Writing a documentation

If you update the code, the documentation should be updated accordingly. 
ETNA uses [Numpydoc style](https://numpydoc.readthedocs.io/en/latest/format.html) for formatting docstrings. 
The documentation is written in ReST.
Length of a line inside docstrings block must be limited to 100 characters to fit into Jupyter documentation popups.

During creation of Pull Request make sure that your documentation looks good, check:
1. `Parameters` and `Returns` sections have correct names and types;
2. Sections should be
   1. divided correctly without artefacts,
   2. consistent by meaning with [Numpydoc Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html);
3. Lists are rendered correctly;
4. Listings of code, e.g. variable names, are typed with monospaced font;
5. Mathematical formulas are rendered correctly;
6. Links to external sources are active;
7. References to python objects should be active if library is listed in [`intersphinx_mapping`](https://github.com/etna-team/etna/blob/master/docs/source/conf.py#L119)

Useful links:
1. [ReST Quickref](https://docutils.sourceforge.io/docs/user/rst/quickref.html)
2. [ReST Roles](https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html)
3. [ReST Cross-referencing Python objects](https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html#cross-referencing-python-objects)
4. [Matplotlib Cheatsheet](https://matplotlib.org/sampledoc/cheatsheet.html)
5. [Sklearn example](https://github.com/scikit-learn/scikit-learn/blob/37ac6788c/sklearn/linear_model/_ridge.py#L321)

The simplest way to check how documentation is rendered is to make a pull request. 
CI will build it, publish and attach a link to the pull request.

#### 6.1 Standard scenarios

**Adding a new method to the class**
- Update the docstrings of the class / method.

**Adding a new public class / function**
- Go to the [`api_reference`](https://github.com/etna-team/etna/tree/master/docs/source/api_reference) directory.
- Find a source page for a relevant module, e.g. `models.rst` is responsible for the `etna.models` module.
- Find a relevant `autosummary` block within the source page and place your new entity there.
  - Make sure you are using the correct `template`. The `class.rst` template should be used for classes, `base.rst` for everything else.
  - Make sure you are using the correct path for the new entity taking into account the `currentmodule` directive.

**Adding a new module**
- Go to the [`api_reference`](https://github.com/etna-team/etna/tree/master/docs/source/api_reference) directory.
- Create a new source page in that directory.

**Adding a new jupyter notebook tutorial**
- Add the notebook to the [`examples`](https://github.com/etna-team/etna/tree/master/examples) directory with its prepended number.
- Add a "launch binder" button to the notebook.
- Add a "Table of contents" for level 2 and 3 headings.
- Install extensions that are necessary for this notebook to run.
- Add imports that aren't related to the topic of the tutorial at the very top.
- Add the new notebook and its table of contents to the `examples/README.md`.
- Add the new notebook to the `README.md`.
- Add a card for the created notebook according to its level of difficulty to [`tutorials.rst`](https://github.com/etna-team/etna/blob/master/docs/source/tutorials.rst).

**Adding a new custom documentation page**
- Create a new page in a [`source`](https://github.com/etna-team/etna/tree/master/docs/source) directory.
- Add a link to the new page to [`user_guide.rst`](https://github.com/etna-team/etna/blob/master/docs/source/user_guide.rst) or any other page responsible for the documentation sections.

#### 6.2 Building locally (optional)

You can also build the documentation locally.
Before building the documentation you may need to install a pandoc package ([pandoc installation guide](https://pandoc.org/installing.html)):
```bash
# Ubuntu
apt-get update && apt-get install -y pandoc

# Mac OS
brew install pandoc

# Windows
choco install pandoc
```

After that, you can try to build the docs with:
```bash
cd docs
make build-docs
```

You could check the result in your browser by opening `build/html/index.html` file.

If you have some issues with building docs - please make sure that you installed the required packages.

## How to make a release

This is only available for the repository members. Steps:
1. Update `docs/source/_static/switcher.json`
  - If you are going to release a new stable version an entry for the current stable version should be created
2. Update `CHANGELOG.md` file:
  - Collect all changes and delete empty bullets
  - Specify version and date of the release
3. Update the version in `pyproject.toml`
4. Create pull request with the changes above
5. Merge the pull request
6. [Create a release](https://github.com/etna-team/etna/releases) with a tag corresponding to a new version 
7. That's all! Our CI/CD will take care of everything else.
