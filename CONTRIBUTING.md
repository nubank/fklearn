# Contributing to fklearn

Hello and Welcome to fklearn!

We love pull requests from everyone.
By participating in this project, you agree to abide by our [code of conduct](CODE-OF-CONDUCT.md).

## Getting Help

* If you found a bug or need a new feature, you can submit an [issue](https://github.com/nubank/fklearn/issues).

* If you would like to chat with other contributors to fklearn, consider joining the 
[![Gitter](https://badges.gitter.im/fklearn-python/general.svg)](https://gitter.im/fklearn-python).

## Contributing

1. Fork, then clone the repo:
    ```bash
    git clone git@github.com:your-username/fklearn.git
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv

    source venv/bin/activate  # For Linux
    ```

3. Install the requirements:
    ```bash
    pip install -qe .[test_deps]
    ```

4. Create a branch for local development:
   ```bash
   git checkout -b name-of-your-bugfix-or-feature
   ```

5. Make your change and add tests for it.

6. Make sure to comply with our coding style.

7. Make sure the tests pass:
    ```bash
    pytest .
    ```

Push to your fork and submit a pull request using our PR template.

At this point you're waiting on us. We will review it according to our internal SLA and we may suggest some changes, 
improvements or alternatives.

## Coding Style

* PEP-8 compliant.
* 120 character line length.

## License

By contributing to fklearn, you agree that your contributions will be licensed under the LICENSE file in the root 
directory of this source tree.
