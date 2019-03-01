from schema import Schema, SchemaError


pipeline_schema = Schema({str: {str: object}})


def validate(schema, spec):
    """
    Parameters
    ----------
    Runs the common validation for all schemas, uses the defined schemas from this
    class init: spec, params and save_data
    schema: schema dict
        The schema we want to validate
    spec: python object
        The spec to be validated
    Returns
    -------
       bool True if we're able to validate the provided spec or False otherwise

    """
    try:
        schema.validate(spec)
    except SchemaError:
        return False
    else:
        return True
