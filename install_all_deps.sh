

REQUIREMENTS_ARR="requirements.txt requirements_catboost.txt requirements_demos.txt requirements_lgbm.txt requirements_tools.txt requirements_xgboost.txt requirements_test.txt"

install_requirements() {
    echo "Installing requirements..."
    for req in $REQUIREMENTS_ARR; do
        echo "Installing $req..."
        pip install -r $req --force-reinstall
    done

    echo "Installing project in development mode..."
    pip install -e . --force-reinstall
}

install_requirements