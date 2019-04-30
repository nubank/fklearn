============
Contributing
============

.. contents:: Table of contents:
   :local:

Where to start?
===============

We love pull requests(and issues) from everyone.
We recommend you to take a look at the project, follow the examples before contribute with code.

By participating in this project, you agree to abide by our code of conduct.

Getting Help
============

If you found a bug or need a new feature, you can submit an `issue <https://github.com/nubank/fklearn/issues>`_.

If you would like to chat with other contributors to fklearn, consider joining the `Gitter <https://gitter.im/fklearn-python>`_.

Working with the code
=====================

Now that that you already understand how the project works, maybe it's time to fix something, add and enhancement, or write new documentation.
It's time to understand how we send contributions.

Version control
---------------

This project is hosted in `Github <https://github.com/nubank/fklearn>`_, this way, to contribute with it you need an account, you sign up `here <https://github.com/signup/free>_`
We use git as version control, so it's good to understand basic git flows before send new code. You can follow `Github Help <https://help.github.com/en>`_ to understand how to work with git.

Fork
----

To write new code, you will iteract with your own fork, so go to `fklearn repo page <https://github.com/nubank/fklearn>`_, and hit the ``Fork`` button. This will create a copy of our repository in your account. To clone the repository in your machine::

    git clone git@github.com:your-username/fklearn.git
    git remote add upstream https://github.com/nubank/fklearn.git

This will create a folder called ``fklearn`` and will connect to the upstream(main repo).

Development env
---------------

We recommend you to create a virtualenv before starts to work with the code.
And be able to run all tests locally before start to write new code.

Creating the virtualenv
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: none

  # Use an ENV_DIR of you choice. We are using ~/venvs
  python3.6 -m venv ~/venvs/fklearn-dev
  source ~/venvs/fklearn-dev/activate

Install the requirements
~~~~~~~~~~~~~~~~~~~~~~~~

This command will install all the test dependencies. To install the package itself follow `install instruction <https://fklearn.readthedocs.io/en/latest/getting_started.html#installation>`_.

.. code-block:: none

  pip install -qe .[test_deps]

Run tests
~~~~~~~~~

The following command should run all tests, if every test pass, you should be ready to start develop new stuff

.. code-block:: none

  python -m pytest tests/

Creating a development branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First we should check if you master is up to date with the latest version of the repo

.. code-block:: none

  git checkout master
  git pull upstream master --ff-only

.. code-block:: none

  git checkout -b name-of-your-bugfix-or-feature

If you already have a branch, and you want to update with the upstream master

.. code-block:: none

  git checkout name-of-your-bugfix-or-feature
  git fetch upstream
  git merge upstream/master

Contribute with code
====================

In this session we'll guide you on how to contribute with the code. This is a guide if you want to fix or implement a new feature.

Code standards
--------------

This project is compatible only with python3.6 and follows the `pep8 style <https://www.python.org/dev/peps/pep-0008/>`_
And we use this `import formatting <https://google.github.io/styleguide/pyguide.html?showone=Imports_formatting#313-imports-formatting>`_

In order to check if your code follow our style, you can run from the repo root dir:

.. code-block:: none

  python -m pip install -q flake8
  python -m flake8 \
  --ignore=E731,W503 \
  --filename=*.py \
  --exclude=__init__.py \
  --show-source \
  --statistics \
  --max-line-length=120 \
  src/ tests/

Run tests
---------

After you finish your feature development or bug fix, you should run your tests, using:


.. code-block:: none

  python -m pytest tests/

Or if want to run only one test:

.. code-block:: none

  python -m pytest tests/test-file-name.py::test_method_name


You should **always** write tests for your features, you can look at the other tests to have a better idea how we implement them.
As test framework we use `pytest <https://docs.pytest.org/en/latest/>`_

Document your code
------------------

All methods should have type annotations, this allow us to know what that method expect as parameters, and what is the output.
You can learn more about it in `typing docs <https://docs.python.org/3.6/library/typing.html>`_

To document your code you should add docstrings, all methods with docstring will appear in this documentation's api file.
If you created a new file, you may need to add it to the ``api.rst`` following the structure

.. code-block:: none

  Folder Name
  -----------

  File name (fklearn.folder_name.file_name)
  #########################################

  ..currentmodule:: fklearn.folder_name.file_name

  .. autosummary::
    method_name

 The docstrings should follow this format

 .. code-block:: none

  """
  Brief introduction of method

  More info about it

  Parameters
  ----------

  parameter_1 : type
      Parameter description

  Returns
  -------

  value_1 : type
      Value description
  """


Contribute with documentation
=============================

You can add, fix documenation of: code(docstrings) or this documentation files.

Docstrings
----------

Follow the same structure we explained in `code contribution <https://fklearn.readthedocs.io/en/latest/contributing.html#document-your-code>`_

Documentation
-------------

This documentation is written using rst(``reStructuredText``) you can learn more about it in `rst docs <http://docutils.sourceforge.net/rst.html>`_
When you make changes in the docs, please make sure, we still be able to build it without any issue.

Build documentation
-------------------

From ``docs/`` folder, install `requirments.txt` and run

.. code-block:: none

  make html

This command will build the documentation inside ``docs/build/html`` and you can check locally how it looks, and if everything worked.

Send you changes to Fklearn repo
================================

Commit your changes
-------------------

You should think about a commit as a unit of change. So it should describe a small change you did in the project.

The following command will list all files you changed:

.. code-block:: none

  git status

To choose which files will be added to the commit:

.. code-block:: none

  git add path/to/the/file/name.extension

And to write a commit message:

This command will open your text editor to write commit messages

.. code-block:: none

  git commit

This will add a commit only with subject

.. code-block:: none

 git commit -m "My commit message"

We recommend this `guide to write better commit messages <https://chris.beams.io/posts/git-commit/>`_

Push the changes
----------------

After you write all your commit messages, decribing what you did, it's time to send to your remote repo.

.. code-block:: none

 git push origin name-of-your-bugfix-or-feature

Create a pull request
---------------------

Now that you already finished your job, you should:
- Go to your repo's Github page
- Click ``New pull request``
- Choose the branch you want to merge
- Review the files that will be merged
- Click ``Create pull request``
- Fill the template
- Tag your PR, add the category(bug, enhancement, documentation...) and a review-request label

When my code will be merged?
----------------------------

All code will be reviewed, we require at least one code owner review, and any other person review.
We will usually do weekly releases of the package if we have any new features, that are already reviewed.

Versioning
==========

Use Semantic versioning to set library versions, more info: `semver.org <https://semver.org/>`_ But basically this means:

1. MAJOR version when you make incompatible API changes,
2. MINOR version when you add functionality in a backwards-compatible manner, and
3. PATCH version when you make backwards-compatible bug fixes.

(from semver.org summary)

You don't need to set the version in your PR, we'll take care of this when we decide to release a new version.
Today the process is:

- Create a new ``milestone`` X.Y.Z (maintainers only)
- Some PR/issues are attibuted to this new milestone
- Merge all the related PRs (maintainers only)
- Create a new PR: ``Bump package to X.Y.Z`` This PR update the version and the change log (maintainers only)
- Create a tag ``X.Y.Z`` (maintainers only)

This last step will trigger the CI to build the package and send the version to pypi

When we add new functionality, the past version will be moved to another branch. For example, if we're at version `1.13.7` and a new functionality is implemented,
we create a new branch `1.13.x`, and protect it(this way we can't delete it), the new code is merged to master branch, and them we create the tag `1.14.0`

This way we can always fix a past version, opening PRs from `1.13.x` branch.
