import os

import jinja2

# Set up the template environment
_template_path = os.path.join(os.path.dirname(__file__), "templates")
_template_loader = jinja2.FileSystemLoader(_template_path)
_template_env = jinja2.Environment(loader=_template_loader, cache_size=-1, trim_blocks=True, lstrip_blocks=True)


def render_template(template_name, **kwargs):
    template = _template_env.get_template(template_name + ".jinja2")
    return template.render(**kwargs)
