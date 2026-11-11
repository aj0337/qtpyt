{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:

{% block modules %}
{%- if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:
{% for item in modules if not item.startswith("test") and (fullname ~ "." ~ item) not in api_skip %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
